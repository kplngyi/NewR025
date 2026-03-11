import argparse
import datetime
import gc
import math
import os
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
import yaml

from braindecode import EEGClassifier
from braindecode.datasets import create_from_mne_raw
from braindecode.util import set_random_seeds
from fe_u import (
    fisher_score_channels_alpha_beta_from_windows_dataset,
    fisher_score_channels_from_windows_dataset_tdpsd,
)
from model_factory import build_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split
from runtime_utils import (
    add_common_runtime_args,
    parse_known_args,
    prepare_runtime_dirs,
    resolve_path,
    resolve_project_root,
)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--data_dir", type=str, default="PPEEG")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument(
    "--model", type=str, default="temporal_se", choices=["shallow", "temporal_se"]
)
parser.add_argument(
    "--fisher_method", type=str, default="tdpsd", choices=["bandpower", "tdpsd"]
)
parser.add_argument(
    "--bandpower_mode", type=str, default="avg", choices=["alpha", "beta", "avg"]
)
parser.add_argument("--top_k", type=int, default=None)
parser.add_argument("--early_stopping_patience", type=int, default=15)
args = parse_known_args(add_common_runtime_args(parser))

# 切换到项目根目录，兼容本地脚本和 Colab 工作目录
cwd = os.getcwd()
project_root = resolve_project_root(__file__, args.project_root)
runtime_dirs = prepare_runtime_dirs(project_root, args.output_root)
print("当前工作目录是：", cwd)
print("项目根目录是：", project_root)
os.chdir(project_root)


target_path = resolve_path(args.data_dir, project_root)
filesA1 = [
    str(target_path / f)
    for f in os.listdir(target_path)
    if f.endswith(".fif") and "A1" in f and not f.startswith(".")
]
filesA1 = sorted(filesA1)
if args.files_limit:
    filesA1 = filesA1[: args.files_limit]
print(filesA1)
if len(filesA1) == 0:
    raise SystemExit(f"No .fif files found in {target_path}")

# 读取配置文件
with open(resolve_path(args.config_path, project_root), "r") as f:
    config = yaml.safe_load(f)
# Keep MNE informational chatter out of training logs.
mne.set_log_level("WARNING")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def resolve_training_device(device_arg):
    requested = (device_arg or "auto").strip().lower()
    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        return torch.device("cuda" if cuda_available else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if requested.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError(
                f"Requested --device {requested} but CUDA is not available."
            )
        try:
            gpu_index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid --device value: {device_arg}") from exc
        gpu_count = torch.cuda.device_count()
        if gpu_index < 0 or gpu_index >= gpu_count:
            raise RuntimeError(
                f"Requested --device {requested}, but only {gpu_count} CUDA device(s) are available."
            )
        return torch.device(requested)
    raise ValueError("Invalid --device. Use one of: auto, cpu, cuda, cuda:N")


def get_fisher_method_tag(method, bandpower_mode):
    if method == "bandpower":
        return f"bandpower_{bandpower_mode}"
    return method


def summarize_training_history(clf, max_epochs, early_stopping_patience):
    history = getattr(clf, "history", None)
    if history is None or len(history) == 0:
        print("No training history available.")
        return

    best_epoch = None
    best_valid_loss = None
    best_valid_accuracy = None

    for row in history:
        valid_loss = row.get("valid_loss")
        if valid_loss is None:
            continue
        if best_valid_loss is None or valid_loss < best_valid_loss:
            best_valid_loss = float(valid_loss)
            best_epoch = int(row.get("epoch", len(history)))
            valid_accuracy = row.get("valid_accuracy", row.get("valid_acc"))
            best_valid_accuracy = (
                None if valid_accuracy is None else float(valid_accuracy)
            )

    epochs_ran = len(history)
    stopped_early = early_stopping_patience > 0 and epochs_ran < max_epochs

    print(f"Epochs completed: {epochs_ran}/{max_epochs}")
    print(f"Early stopping triggered: {stopped_early}")
    if best_epoch is not None:
        print(f"Best valid_loss epoch: {best_epoch}")
        print(f"Best valid_loss: {best_valid_loss:.4f}")
        if best_valid_accuracy is not None:
            print(f"Best-epoch valid_accuracy: {best_valid_accuracy:.4f}")


EVENT_LABELS = ("Rest", "Elbow_Flexion", "Elbow_Extension")
EVENT_MAPPING = {"Rest": 0, "Elbow_Flexion": 1, "Elbow_Extension": 2}


def build_trial_records(
    raw, matched_onsets, rest_duration, flex_duration, ext_duration
):
    sfreq = float(raw.info["sfreq"])
    raw_end_time = raw.times[-1] + (1.0 / sfreq)
    trial_records = []
    prev_ext_end = None

    for trial_id, flex_time in enumerate(matched_onsets):
        rest_onset = (
            flex_time - rest_duration
            if prev_ext_end is None
            else min(prev_ext_end, flex_time - rest_duration)
        )
        flex_onset = flex_time
        ext_onset = flex_time + flex_duration
        trial_end = ext_onset + ext_duration
        prev_ext_end = trial_end

        if rest_onset < 0 or trial_end > raw_end_time:
            print(
                f"Skipping trial {trial_id}: outside raw bounds "
                f"(start={rest_onset:.3f}, end={trial_end:.3f}, raw_end={raw_end_time:.3f})."
            )
            continue

        crop_end = max(rest_onset, trial_end - (1.0 / sfreq))
        trial_raw = raw.copy().crop(tmin=rest_onset, tmax=crop_end)
        trial_raw.set_annotations(
            mne.Annotations(
                onset=[0.0, flex_onset - rest_onset, ext_onset - rest_onset],
                duration=[rest_duration, flex_duration, ext_duration],
                description=list(EVENT_LABELS),
            )
        )
        trial_records.append(
            {
                "trial_id": trial_id,
                "raw": trial_raw,
            }
        )

    return trial_records


def split_trial_records(trial_records, random_state=710):
    if len(trial_records) < 4:
        raise ValueError(
            f"Need at least 4 trials for a 70/15/15 split, got {len(trial_records)}"
        )

    indices = np.arange(len(trial_records))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=random_state
    )
    valid_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=random_state
    )

    return (
        [trial_records[i] for i in train_idx],
        [trial_records[i] for i in valid_idx],
        [trial_records[i] for i in test_idx],
    )


def create_windows_dataset_from_trials(
    trial_records, subject_id, window_size_samples, window_stride_samples
):
    if not trial_records:
        return []

    parts = [record["raw"] for record in trial_records]
    descriptions = [
        {
            "event_code": [0, 1, 2],
            "subject": subject_id,
            "trial_id": int(record["trial_id"]),
        }
        for record in trial_records
    ]
    return create_from_mne_raw(
        parts,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False,
        descriptions=descriptions,
        mapping=EVENT_MAPPING,
    )


def select_windows_with_channels(windows_dataset, selected_channels):
    selected_samples = []
    for i in range(len(windows_dataset)):
        X_i, y_i, meta_i = windows_dataset[i]
        X_i = np.array(X_i)
        X_i_sel = X_i[selected_channels, :]
        selected_samples.append((X_i_sel, int(y_i), meta_i))
    return selected_samples


# ------------------ 全局超参数（可按需调整） ------------------

rest_dur = 20
flex_dur = 10
ext_dur = 5

# lr = 0.001              # 初始学习率，不宜太大，EEG 数据容易震荡
# weight_decay = 1e-4     # 轻微权重衰减，有助于正则化
# batch_size = 32         # EEG 样本通常不大，32-64都可以，训练速度和稳定性平衡
# n_epochs = 100         # 训练轮数稍多，让学习率退火充分发挥作用

# 超参数（直接从 config 里取）
MAX_CHANNELS = 63
MIN_TOP_K = math.ceil(MAX_CHANNELS * 0.1)
if args.min_top_k is not None:
    MIN_TOP_K = args.min_top_k
TOP_K_STEP = args.top_k_step
if TOP_K_STEP <= 0:
    raise ValueError(f"--top_k_step must be a positive integer, got {TOP_K_STEP}")

requested_top_k = (
    args.top_k if args.top_k is not None else config.get("top_k", MAX_CHANNELS)
)
if requested_top_k is None:
    requested_top_k = MAX_CHANNELS
if requested_top_k <= 0:
    raise ValueError(f"top_k must be a positive integer, got {requested_top_k}")
top_k = min(int(requested_top_k), MAX_CHANNELS)
if requested_top_k > MAX_CHANNELS:
    print(
        f"Requested top_k {requested_top_k} exceeds MAX_CHANNELS={MAX_CHANNELS}; "
        f"clamping to {top_k}."
    )

window_size_samples = config["window_size_samples"]
window_stride_samples = config["window_stride_samples"]
batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
n_epochs = args.epochs if args.epochs is not None else config["n_epochs"]
lr = config["lr"]
weight_decay = config["weight_decay"]
seed = config["seed"]
fisher_method_tag = get_fisher_method_tag(args.fisher_method, args.bandpower_mode)

# ------------------ 主循环 ------------------
while top_k >= MIN_TOP_K:
    global_results = []  # 列表，后面会 append dicts: {'subject':..., 'top_k':..., 'test_acc':..., ...}
    # ------------------ 输出重定向（安全） ------------------
    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = runtime_dirs["logs_dir"]
    save_dir = runtime_dirs["output_dir"] / "ResEEG"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    out_fname = os.path.join(
        log_dir, f"{now_time}_{fisher_method_tag}_{top_k}_{n_epochs}_{lr}_traineeg.log"
    )
    orig_stdout = sys.stdout
    f_out = open(out_fname, "w")
    sys.stdout = Tee(orig_stdout, f_out)
    # 打印超参数
    print("\n📌 Training Hyperparameters:")
    print(f"保留的通道数: {top_k}")
    print(f"窗口大小: {window_size_samples} 样本")
    print(f"窗口步长: {window_stride_samples} 样本")
    print(f"学习率: {lr}")
    print(f"权重衰减: {weight_decay}")
    print(f"批大小: {batch_size}")
    print(f"训练轮数: {n_epochs}")
    print(f"模型: {args.model}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"模态最大通道数: {MAX_CHANNELS}")
    print(f"最小搜索通道数: {MIN_TOP_K}")
    print(f"通道搜索步长: {TOP_K_STEP}")
    print(f"Fisher方法: {fisher_method_tag}")
    try:
        print("Files to process:", filesA1)
        # ------------------ 处理每个 fif 文件循环 ------------------
        # 你需要在外部定义 filesA1 = [...] 列表（fif 文件路径）

        for filea1 in filesA1:
            resname = os.path.basename(filea1[:-4])
            subject_id = resname[:4]
            print("Processing file:", resname)
            print("subject_id:", subject_id)

            # 读取 raw（copy 避免原始对象被修改）
            raw = mne.io.read_raw_fif(filea1, preload=True)
            raw = raw.copy()

            # 通过注释创建事件（你原有逻辑）
            events, event_id = mne.events_from_annotations(raw)
            print("Original event_id:", event_id)
            # 你的文件中目标注释描述
            target_desc = "Stimulus/S  5"
            annotations = raw.annotations
            matched_onsets = [
                onset
                for onset, desc in zip(annotations.onset, annotations.description)
                if desc.strip() == target_desc.strip()
            ]
            print("Matched onsets for 'Stimulus/S  5':", matched_onsets)
            if len(matched_onsets) == 0:
                print("No matched onsets found, skipping file.")
                continue
            n_trials = len(matched_onsets)
            print(f"n_trials: {n_trials}")

            trial_records = build_trial_records(
                raw,
                matched_onsets,
                rest_duration=rest_dur,
                flex_duration=flex_dur,
                ext_duration=ext_dur,
            )
            print("Retained trial count:", len(trial_records))
            if len(trial_records) < 4:
                print(
                    "Not enough complete trials for a 70/15/15 trial split, skipping file."
                )
                continue

            try:
                train_trials, valid_trials, test_trials = split_trial_records(
                    trial_records, random_state=710
                )
            except ValueError as exc:
                print(f"Trial split failed: {exc}")
                continue

            print(
                "Trial split sizes:",
                len(train_trials),
                len(valid_trials),
                len(test_trials),
            )

            windows_dataset = create_windows_dataset_from_trials(
                trial_records,
                subject_id=subject_id,
                window_size_samples=window_size_samples,
                window_stride_samples=window_stride_samples,
            )
            print("Created full windows_dataset, length:", len(windows_dataset))
            if len(windows_dataset) == 0:
                print("No windows created from retained trials, skipping file.")
                continue

            if args.fisher_method == "bandpower":
                fs = float(raw.info["sfreq"])
                rank_idx, channel_scores = (
                    fisher_score_channels_alpha_beta_from_windows_dataset(
                        windows_dataset,
                        fs=fs,
                        mode=args.bandpower_mode,
                    )
                )
            else:
                rank_idx, channel_scores, _ = (
                    fisher_score_channels_from_windows_dataset_tdpsd(windows_dataset)
                )

            n_channels_total = np.array(windows_dataset[0][0]).shape[0]
            top_k_use = min(top_k, n_channels_total)
            selected_channels = list(rank_idx[:top_k_use])
            print(f"Total channels: {n_channels_total}, selecting top_k = {top_k_use}")
            print(f"Channel selection fisher method: {fisher_method_tag}")
            print("Selected channel indices:", selected_channels)
            print("Selected channel scores:", channel_scores[selected_channels])

            train_windows_dataset = create_windows_dataset_from_trials(
                train_trials,
                subject_id=subject_id,
                window_size_samples=window_size_samples,
                window_stride_samples=window_stride_samples,
            )
            valid_windows_dataset = create_windows_dataset_from_trials(
                valid_trials,
                subject_id=subject_id,
                window_size_samples=window_size_samples,
                window_stride_samples=window_stride_samples,
            )
            test_windows_dataset = create_windows_dataset_from_trials(
                test_trials,
                subject_id=subject_id,
                window_size_samples=window_size_samples,
                window_stride_samples=window_stride_samples,
            )

            if (
                min(
                    len(train_windows_dataset),
                    len(valid_windows_dataset),
                    len(test_windows_dataset),
                )
                == 0
            ):
                print("One of the split datasets produced no windows, skipping file.")
                continue

            train_set = select_windows_with_channels(
                train_windows_dataset, selected_channels
            )
            valid_set = select_windows_with_channels(
                valid_windows_dataset, selected_channels
            )
            test_set = select_windows_with_channels(
                test_windows_dataset, selected_channels
            )

            def extract_X_y_from_sample_list(sample_list):
                X_list = []
                y_list = []
                for X, y, _ in sample_list:
                    X_list.append(np.array(X))
                    y_list.append(y)
                X_all = np.stack(X_list)  # (n_trials, n_ch_sel, n_time)
                y_all = np.array(y_list)
                return X_all, y_all

            X_train, y_train = extract_X_y_from_sample_list(train_set)
            X_valid, y_valid = extract_X_y_from_sample_list(valid_set)
            X_test, y_test = extract_X_y_from_sample_list(test_set)

            print(
                "Train/Valid/Test sizes:",
                X_train.shape[0],
                X_valid.shape[0],
                X_test.shape[0],
            )
            print(
                "Shapes (n_trials, n_ch_sel, n_time):",
                X_train.shape,
                X_valid.shape,
                X_test.shape,
            )
            print("Selected channel count used for training:", X_train.shape[1])

            # ------------------ 设置随机种子与设备 ------------------
            train_device = resolve_training_device(args.device)
            use_cuda = train_device.type == "cuda"
            if use_cuda:
                torch.backends.cudnn.benchmark = True
            set_random_seeds(seed=seed, cuda=use_cuda)
            print(f"Requested device: {args.device}")
            print(f"Resolved device: {train_device}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if use_cuda:
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Using GPU: {torch.cuda.get_device_name(train_device)}")

            # ------------------ 构建模型（注意 n_channels 来自 selected channels） ------------------
            n_channels = X_train.shape[1]
            input_window_samples = X_train.shape[2]
            classes = np.unique(y_train)
            n_classes = len(classes)

            model = build_model(
                model_name=args.model,
                n_channels=n_channels,
                n_classes=n_classes,
                n_times=input_window_samples,
            )
            print(f"Model: {args.model}")
            print(model)

            # ------------------ 损失函数：根据训练集计算 class weights ------------------
            class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weights = torch.FloatTensor(class_weights).to(train_device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

            callbacks = [
                "accuracy",
                (
                    "lr_scheduler",
                    LRScheduler("CosineAnnealingLR", T_max=max(1, n_epochs - 1)),
                ),
            ]
            if args.early_stopping_patience > 0:
                callbacks.append(
                    (
                        "early_stopping",
                        EarlyStopping(
                            monitor="valid_loss",
                            patience=args.early_stopping_patience,
                            lower_is_better=True,
                        ),
                    )
                )

            # ------------------ 构建 EEGClassifier 并训练 ------------------
            clf = EEGClassifier(
                model,
                criterion=criterion,
                optimizer=torch.optim.AdamW,
                train_split=predefined_split(valid_set),
                optimizer__lr=lr,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                callbacks=callbacks,
                device=str(train_device),
                classes=classes,
                max_epochs=n_epochs,
            )

            print("Start training...")
            clf.fit(X_train, y=y_train)
            print("Training finished.")
            summarize_training_history(
                clf,
                max_epochs=n_epochs,
                early_stopping_patience=args.early_stopping_patience,
            )

            # ------------------ 评估 ------------------
            y_pred_test = clf.predict(X_test)
            test_acc = clf.score(X_test, y=y_test)
            print(f"Test acc: {(test_acc * 100):.2f}%")

            labels_for_cm = clf.classes_
            cm_test = confusion_matrix(y_test, y_pred_test, labels=labels_for_cm)

            # ---------- 保存结果 ----------
            def plot_and_save(cm, labels, title, fname):
                disp = ConfusionMatrixDisplay(cm, display_labels=labels)
                fig, ax = plt.subplots(figsize=(4, 4))
                disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
                ax.set_title(title)
                fig.tight_layout()
                fig.savefig(fname, dpi=300)
                plt.close(fig)
                print(f"Saved image: {fname}")

            plot_and_save(
                cm_test,
                labels_for_cm,
                "Confusion Matrix (Test)",
                os.path.join(
                    save_dir,
                    f"{resname}_{fisher_method_tag}_{batch_size}_{n_epochs}_{top_k}_{lr}_cm_test.png",
                ),
            )
            pd.DataFrame(cm_test, index=labels_for_cm, columns=labels_for_cm).to_csv(
                os.path.join(
                    save_dir,
                    f"{resname}_{fisher_method_tag}_{batch_size}_{n_epochs}_{top_k}_{lr}_cm_test.csv",
                )
            )
            print("CSV 已保存到", save_dir)

            # 记录所选通道到文件，便于审查
            pd.DataFrame(
                {
                    "selected_channel_idx": selected_channels,
                    "score": channel_scores[selected_channels],
                    "fisher_method": fisher_method_tag,
                }
            ).to_csv(
                os.path.join(
                    save_dir,
                    f"{resname}_{fisher_method_tag}_{top_k}_selected_channels.csv",
                ),
                index=False,
            )
            print("Selected channels saved.")

            # ---------- 将结果记录到 global_results（用于后续比较） ----------
            try:
                ch_names = raw.info.get("ch_names", None)
                selected_channel_names = (
                    [ch_names[i] for i in selected_channels]
                    if ch_names is not None
                    else []
                )
            except Exception:
                selected_channel_names = []

            result_entry = {
                "subject": subject_id,
                "file": resname,
                "top_k": int(top_k_use),
                "n_channels_total": int(n_channels_total),
                "n_channels_selected": int(len(selected_channels)),
                "n_train": int(X_train.shape[0]),
                "n_valid": int(X_valid.shape[0]),
                "n_test": int(X_test.shape[0]),
                "test_acc": float(test_acc),
                "fisher_method": fisher_method_tag,
                "selected_channel_idx": selected_channels,
                "selected_channel_scores": [
                    float(s) for s in channel_scores[selected_channels]
                ],
                "all_channel_scores": [float(s) for s in channel_scores],
                "selected_channel_names": selected_channel_names,
            }
            global_results.append(result_entry)

            # ------------------ 清理释放内存 / GPU ------------------
            del clf, model
            del X_train, y_train, X_valid, y_valid, X_test, y_test
            del train_set, valid_set, test_set
            del (
                train_windows_dataset,
                valid_windows_dataset,
                test_windows_dataset,
                windows_dataset,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            gc.collect()

    finally:
        sys.stdout = orig_stdout
        f_out.close()
        print(f"Script finished. Log saved to {out_fname}")

    # 保存summary csv
    summary_df = pd.DataFrame(global_results)
    summary_csv = os.path.join(
        save_dir,
        f"summary_{fisher_method_tag}_{n_epochs}_{batch_size}_{top_k}_results.csv",
    )
    summary_df.to_csv(summary_csv, index=False)
    print("Saved summary CSV:", summary_csv)
    top_k = top_k - TOP_K_STEP
