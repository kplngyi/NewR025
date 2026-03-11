import argparse
import datetime
import gc
import math
import os
import traceback

import matplotlib
import mne
import numpy as np
import pandas as pd
import torch
import yaml

from braindecode import EEGClassifier
from braindecode.datasets import create_from_mne_raw
from braindecode.util import set_random_seeds
from model_factory import build_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from skorch.callbacks import LRScheduler
from skorch.dataset import Dataset as SkorchDataset
from skorch.helper import predefined_split
from runtime_utils import (
    add_common_runtime_args,
    parse_known_args,
    prepare_runtime_dirs,
    resolve_path,
    resolve_project_root,
)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--data_dir", type=str, default="FusionEEG-fNIRS")
parser.add_argument(
    "--model", type=str, default="shallow", choices=["shallow", "temporal_se"]
)
parser.add_argument("--top_k", type=int, default=None)
args = parse_known_args(add_common_runtime_args(parser))


# 切换到项目根目录，兼容本地脚本和 Colab 工作目录
cwd = os.getcwd()
project_root = resolve_project_root(__file__, args.project_root)
runtime_dirs = prepare_runtime_dirs(project_root, args.output_root)
print("当前工作目录是：", cwd)
print("项目根目录是：", project_root)
os.chdir(project_root)

# 路径设置
fusion_dir = resolve_path(args.data_dir, project_root)

# 获取所有 EEG 和 fNIRS 文件名
fusion_files = os.listdir(fusion_dir)
print(f"✅ 找到 Fusion(EEG + fNIRS) 数据: {fusion_files}")

# 提取被试（如 N008_A1L）
fusion_keys = set(
    f.split("_eeg_fnirs")[0] for f in fusion_files if f.endswith("_eeg_fnirs_raw.fif")
)

# 取交集，保证匹配
matched_keys = sorted(fusion_keys)
if args.files_limit:
    matched_keys = matched_keys[: args.files_limit]

print(f"✅ 找到匹配的 Fusion(EEG + fNIRS) 数据: {matched_keys}")
print(f"共 {int(len(matched_keys) / 2)} 个被试")
if len(matched_keys) == 0:
    raise SystemExit(f"No fusion files found in {fusion_dir}")

import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


EVENT_LABELS = ("Rest", "Elbow_Flexion", "Elbow_Extension")
EVENT_MAPPING = {"Rest": 0, "Elbow_Flexion": 1, "Elbow_Extension": 2}


def build_trial_records_from_annotations(raw):
    sfreq = float(raw.info["sfreq"])
    raw_end_time = raw.times[-1] + (1.0 / sfreq)
    annotations = raw.annotations
    triplet = list(EVENT_LABELS)
    trial_records = []

    descriptions = [str(desc).strip() for desc in annotations.description]
    onsets = list(annotations.onset)
    durations = list(annotations.duration)

    for idx in range(0, len(descriptions) - 2, 3):
        current_labels = descriptions[idx : idx + 3]
        if current_labels != triplet:
            print(
                f"Skipping annotation block starting at index {idx}: labels={current_labels}"
            )
            continue

        rest_onset = float(onsets[idx])
        flex_onset = float(onsets[idx + 1])
        ext_onset = float(onsets[idx + 2])
        rest_duration = float(durations[idx])
        flex_duration = float(durations[idx + 1])
        ext_duration = float(durations[idx + 2])
        trial_end = ext_onset + ext_duration

        if rest_onset < 0 or trial_end > raw_end_time:
            print(
                f"Skipping trial {len(trial_records)}: outside raw bounds "
                f"(start={rest_onset:.3f}, end={trial_end:.3f}, raw_end={raw_end_time:.3f})."
            )
            continue

        crop_end = max(rest_onset, trial_end - (1.0 / sfreq))
        trial_raw = raw.copy().crop(tmin=rest_onset, tmax=crop_end)
        trial_raw.set_annotations(
            mne.Annotations(
                onset=[0.0, flex_onset - rest_onset, ext_onset - rest_onset],
                duration=[rest_duration, flex_duration, ext_duration],
                description=triplet,
            )
        )
        trial_records.append({"trial_id": len(trial_records), "raw": trial_raw})

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


def fisher_score_channels_from_windows_dataset(windows_dataset):
    """
    计算 Fisher score（每窗口每通道 mean(abs(X)) 作为通道特征）。
    windows_dataset[i] -> (X, y, meta)，X shape = (n_ch, n_time)
    返回 rank_idx（按分数从大到小的通道索引）和 scores（每通道分数）
    """
    n_windows = len(windows_dataset)
    if n_windows == 0:
        raise ValueError("windows_dataset is empty")
    sample_X, sample_y, _ = windows_dataset[0]
    sample_X = np.array(sample_X)
    n_channels = sample_X.shape[0]
    F = np.zeros((n_windows, n_channels), dtype=float)
    ys = np.zeros((n_windows,), dtype=int)
    for i in range(n_windows):
        X_i, y_i, _ = windows_dataset[i]
        X_i = np.array(X_i)
        if X_i.shape[0] != n_channels:
            raise ValueError(
                f"Inconsistent channel count at window {i}: {X_i.shape[0]} vs {n_channels}"
            )
        F[i, :] = np.mean(np.abs(X_i), axis=1)
        ys[i] = int(y_i)
    classes = np.unique(ys)
    mu_total = F.mean(axis=0)
    Sb = np.zeros(n_channels, dtype=float)
    Sw = np.zeros(n_channels, dtype=float)
    for c in classes:
        Xc = F[ys == c]
        nc = Xc.shape[0]
        if nc == 0:
            continue
        muc = Xc.mean(axis=0)
        varc = Xc.var(axis=0)
        Sb += nc * (muc - mu_total) ** 2
        Sw += nc * varc
    scores = Sb / (Sw + 1e-8)
    rank_idx = np.argsort(scores)[::-1]
    return rank_idx, scores


def extract_X_y_from_sample_list(sample_list):
    X_list = []
    y_list = []
    for X, y, _ in sample_list:
        X_list.append(np.array(X))
        y_list.append(int(y))
    X_all = np.stack(X_list)  # (n_trials, n_ch_sel, n_time)
    y_all = np.array(y_list)
    return X_all, y_all


def plot_and_save(cm, labels, title, fname):
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)


# --------------- 可配置参数 ---------------

# 超参数（直接从 config 里取）
MAX_CHANNELS = 63 + 88
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
# ------------------ 主循环 ------------------

while top_k >= MIN_TOP_K:
    global_results = []  # 列表，后面会 append dicts: {'subject':..., 'top_k':..., 'test_acc':..., ...}
    # ------------------ 输出重定向（安全） ------------------
    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = runtime_dirs["logs_dir"]
    os.makedirs(log_dir, exist_ok=True)
    out_fname = os.path.join(
        log_dir, f"{now_time}_{top_k}_{n_epochs}_{lr}_trainfusion.log"
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
    print(f"模态最大通道数: {MAX_CHANNELS}")
    print(f"最小搜索通道数: {MIN_TOP_K}")
    print(f"通道搜索步长: {TOP_K_STEP}")

    save_dir = runtime_dirs["output_dir"] / "ResFusion"
    os.makedirs(save_dir, exist_ok=True)

    try:
        print("Start processing. Log ->", out_fname)
        for key in matched_keys:
            try:
                fusion_path = os.path.join(fusion_dir, key + "_eeg_fnirs_raw.fif")
                print("\n🔄 Processing:", fusion_path)
                if not os.path.exists(fusion_path):
                    print("File not found, skipping:", fusion_path)
                    continue

                # 读取原始数据
                raw = mne.io.read_raw_fif(fusion_path, preload=True)
                raw = raw.copy()

                # 配置窗口参数
                window_size = window_size_samples
                window_stride = window_stride_samples

                subject_id = key[:4]

                trial_records = build_trial_records_from_annotations(raw)
                print("Retained trial count:", len(trial_records))
                if len(trial_records) < 4:
                    print(
                        "Not enough complete trials for a 70/15/15 trial split, skipping."
                    )
                    continue

                train_trials, valid_trials, test_trials = split_trial_records(
                    trial_records, random_state=710
                )
                print(
                    "Trial split sizes:",
                    len(train_trials),
                    len(valid_trials),
                    len(test_trials),
                )

                windows_dataset = create_windows_dataset_from_trials(
                    trial_records,
                    subject_id=subject_id,
                    window_size_samples=window_size,
                    window_stride_samples=window_stride,
                )

                print("Created full windows_dataset, length:", len(windows_dataset))
                if len(windows_dataset) == 0:
                    print("No windows created from retained trials, skipping.")
                    continue

                # ---------- 通道选择（Fisher score） ----------
                rank_idx, channel_scores = fisher_score_channels_from_windows_dataset(
                    windows_dataset
                )
                n_channels_total = np.array(windows_dataset[0][0]).shape[0]
                top_k_use = min(top_k, n_channels_total)
                selected_channels = list(rank_idx[:top_k_use])
                print(
                    f"Total channels: {n_channels_total}, selecting top_k = {top_k_use}"
                )
                print("Selected channel indices:", selected_channels)
                print("Selected channel scores:", channel_scores[selected_channels])

                # 保存所选通道信息（若可用）
                try:
                    ch_names = raw.info.get("ch_names", None)
                    if ch_names is not None:
                        selected_channel_names = [
                            ch_names[i] for i in selected_channels
                        ]
                        pd.DataFrame(
                            {
                                "idx": selected_channels,
                                "name": selected_channel_names,
                                "score": channel_scores[selected_channels],
                            }
                        ).to_csv(f"{save_dir}/{key}_selected_channels.csv", index=False)
                        print("Saved selected channel info to CSV.")
                except Exception as e:
                    print("Warning saving channel names:", e)

                train_windows_dataset = create_windows_dataset_from_trials(
                    train_trials,
                    subject_id=subject_id,
                    window_size_samples=window_size,
                    window_stride_samples=window_stride,
                )
                valid_windows_dataset = create_windows_dataset_from_trials(
                    valid_trials,
                    subject_id=subject_id,
                    window_size_samples=window_size,
                    window_stride_samples=window_stride,
                )
                test_windows_dataset = create_windows_dataset_from_trials(
                    test_trials,
                    subject_id=subject_id,
                    window_size_samples=window_size,
                    window_stride_samples=window_stride,
                )

                if (
                    min(
                        len(train_windows_dataset),
                        len(valid_windows_dataset),
                        len(test_windows_dataset),
                    )
                    == 0
                ):
                    print("One of the split datasets produced no windows, skipping.")
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

                # dtype 强制转换
                X_train = X_train.astype(np.float32)
                X_valid = X_valid.astype(np.float32)
                X_test = X_test.astype(np.float32)
                y_train = y_train.astype(np.int64)
                y_valid = y_valid.astype(np.int64)
                y_test = y_test.astype(np.int64)

                # ------------------ 随机种子与设备 ------------------
                cuda = torch.cuda.is_available()
                device = "cuda" if cuda else "cpu"
                if cuda:
                    torch.backends.cudnn.benchmark = False
                set_random_seeds(seed=seed, cuda=cuda)

                # ------------------ 构建模型 ------------------
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
                if cuda:
                    model.cuda()

                # ------------------ 损失函数与 class weights ------------------
                class_weights = compute_class_weight(
                    "balanced", classes=classes, y=y_train
                )
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                    device
                )
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

                # ------------------ Skorch Dataset for valid ------------------
                valid_ds = SkorchDataset(X_valid, y_valid)

                # ------------------ 构建 EEGClassifier ------------------
                clf = EEGClassifier(
                    model,
                    criterion=criterion,
                    optimizer=torch.optim.AdamW,
                    train_split=predefined_split(valid_ds),
                    optimizer__lr=lr,
                    optimizer__weight_decay=weight_decay,
                    batch_size=batch_size,
                    callbacks=[
                        "accuracy",
                        (
                            "lr_scheduler",
                            LRScheduler(
                                "CosineAnnealingLR", T_max=max(1, n_epochs - 1)
                            ),
                        ),
                    ],
                    device=device,
                    classes=classes,
                    max_epochs=n_epochs,
                )

                print("Start training...")
                clf.fit(X_train, y=y_train)
                print("Training finished.")

                # ---------- 评估 ----------
                y_pred_test = clf.predict(X_test)
                test_acc = clf.score(X_test, y=y_test)
                print(f"Test acc: {(test_acc * 100):.2f}%")

                labels_for_cm = clf.classes_
                cm_test = confusion_matrix(y_test, y_pred_test, labels=labels_for_cm)

                # ---------- 保存结果 ----------
                resname = f"{key}_"
                plot_and_save(
                    cm_test,
                    labels_for_cm,
                    "Confusion Matrix (Test)",
                    f"{save_dir}/{resname}_{batch_size}_{n_epochs}_{top_k}_{lr}_cm_test.png",
                )
                pd.DataFrame(
                    cm_test, index=labels_for_cm, columns=labels_for_cm
                ).to_csv(
                    f"{save_dir}/{resname}_{batch_size}_{n_epochs}_{top_k}_{lr}_cm_test.csv"
                )
                print("Saved results to", save_dir)
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
                    "selected_channel_idx": selected_channels,
                    "selected_channel_names": selected_channel_names,
                }
                global_results.append(result_entry)
                # ---------- 清理 ----------
                del clf, model
                del X_train, y_train, X_valid, y_valid, X_test, y_test
                del train_set, valid_set, test_set
                del (
                    train_windows_dataset,
                    valid_windows_dataset,
                    test_windows_dataset,
                    windows_dataset,
                )
                del rank_idx, channel_scores, selected_channels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                gc.collect()

            except Exception as e:
                print(f"Error processing key {key}: {e}")
                traceback.print_exc()
                # 尝试释放并继续下一个文件/键
                try:
                    if "model" in locals():
                        del model
                    if "clf" in locals():
                        del clf
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    pass
                continue

    finally:
        # 恢复 stdout 并关闭文件
        sys.stdout = orig_stdout
        f_out.close()
        print("Processing finished. Log saved to", out_fname)
    ## 保存csv
    summary_df = pd.DataFrame(global_results)
    summary_csv = os.path.join(
        save_dir, f"summary_{n_epochs}_{batch_size}_{top_k}_results.csv"
    )
    summary_df.to_csv(summary_csv, index=False)
    print("Saved summary CSV:", summary_csv)
    top_k = top_k - TOP_K_STEP
