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
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import compute_class_weight
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import Dataset as SkorchDataset
from skorch.helper import predefined_split
from runtime_utils import (
    add_common_runtime_args,
    create_balanced_sampler,
    normalize_channel_scores,
    parse_known_args,
    prepare_runtime_dirs,
    resolve_requested_top_k,
    resolve_path,
    resolve_project_root,
    SampledClassRatioLogger,
)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--data_dir", type=str, default="FusionEEG-fNIRS")
parser.add_argument(
    "--model", type=str, default="shallow", choices=["shallow", "temporal_se"]
)
parser.add_argument("--top_k", type=int, default=None)
parser.add_argument("--early_stopping_patience", type=int, default=None)
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


def build_trial_splits(
    trial_records, use_cross_validation=False, cv_folds=5, random_state=710
):
    if not use_cross_validation:
        train_trials, valid_trials, test_trials = split_trial_records(
            trial_records, random_state=random_state
        )
        return [
            {
                "split_name": "holdout",
                "fold_id": None,
                "train_trials": train_trials,
                "valid_trials": valid_trials,
                "test_trials": test_trials,
            }
        ]

    if cv_folds < 3:
        raise ValueError(f"cv_folds must be at least 3, got {cv_folds}")
    if len(trial_records) < cv_folds:
        raise ValueError(
            f"Need at least {cv_folds} trials for {cv_folds}-fold cross-validation, got {len(trial_records)}"
        )

    indices = np.arange(len(trial_records))
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_indices = [indices[test_idx] for _, test_idx in kfold.split(indices)]
    split_defs = []
    for fold_idx in range(cv_folds):
        test_idx = fold_indices[fold_idx]
        valid_idx = fold_indices[(fold_idx + 1) % cv_folds]
        train_parts = [
            fold_indices[i]
            for i in range(cv_folds)
            if i not in {fold_idx, (fold_idx + 1) % cv_folds}
        ]
        train_idx = (
            np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
        )
        split_defs.append(
            {
                "split_name": f"fold{fold_idx + 1}",
                "fold_id": int(fold_idx + 1),
                "train_trials": [trial_records[i] for i in train_idx],
                "valid_trials": [trial_records[i] for i in valid_idx],
                "test_trials": [trial_records[i] for i in test_idx],
            }
        )
    return split_defs


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


def summarize_training_history(clf, max_epochs, early_stopping_patience):
    history = getattr(clf, "history", None)
    if history is None or len(history) == 0:
        print("No training history available.")
        return

    best_epoch = None
    best_valid_loss = None
    best_valid_accuracy = None
    best_valid_f1_macro = None
    best_valid_balanced_accuracy = None

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
            valid_f1_macro = row.get("valid_f1_macro")
            best_valid_f1_macro = (
                None if valid_f1_macro is None else float(valid_f1_macro)
            )
            valid_balanced_accuracy = row.get("valid_balanced_accuracy")
            best_valid_balanced_accuracy = (
                None
                if valid_balanced_accuracy is None
                else float(valid_balanced_accuracy)
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
        if best_valid_f1_macro is not None:
            print(f"Best-epoch valid_f1_macro: {best_valid_f1_macro:.4f}")
        if best_valid_balanced_accuracy is not None:
            print(
                "Best-epoch valid_balanced_accuracy: "
                f"{best_valid_balanced_accuracy:.4f}"
            )


def monitor_lower_is_better(monitor_name):
    if monitor_name == "valid_loss":
        return True
    if monitor_name in {
        "valid_accuracy",
        "valid_acc",
        "valid_f1_macro",
        "valid_balanced_accuracy",
    }:
        return False
    raise ValueError(f"Unsupported early_stop_monitor: {monitor_name}")


def save_cv_summary_files(summary_df, save_dir, prefix):
    if summary_df.empty:
        return

    metric_columns = [
        "test_acc",
        "test_f1_macro",
        "test_balanced_accuracy",
    ]
    for group_name, output_name in [
        (["subject", "file"], f"{prefix}_per_file_cv_summary.csv"),
        (["subject"], f"{prefix}_per_subject_cv_summary.csv"),
    ]:
        if not all(column in summary_df.columns for column in group_name):
            continue
        grouped = summary_df.groupby(group_name, dropna=False)
        aggregated = grouped[metric_columns].agg(["mean", "std", "min", "max"])
        aggregated.columns = ["_".join(col).strip("_") for col in aggregated.columns]
        aggregated = aggregated.reset_index()
        aggregated["n_splits"] = grouped.size().values
        aggregated.to_csv(os.path.join(save_dir, output_name), index=False)
        print("Saved CV summary CSV:", os.path.join(save_dir, output_name))


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
    scores_norm = normalize_channel_scores(scores)
    rank_idx = np.argsort(scores_norm)[::-1]
    return rank_idx, scores, scores_norm


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

top_k = resolve_requested_top_k(args.top_k, config.get("top_k"), MAX_CHANNELS)
requested_top_k = top_k
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
early_stopping_patience = (
    args.early_stopping_patience
    if args.early_stopping_patience is not None
    else int(config.get("early_stop_patience", 15))
)
early_stopping_monitor = str(config.get("early_stop_monitor", "valid_loss"))
early_stopping_threshold = float(config.get("early_stop_threshold", 1e-4))
early_stopping_lower_is_better = monitor_lower_is_better(early_stopping_monitor)
use_cross_validation = bool(config.get("use_cross_validation", False))
cv_folds = int(config.get("cv_folds", 5))
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
    print(f"模型: {args.model}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Early stopping monitor: {early_stopping_monitor}")
    print(f"Early stopping threshold: {early_stopping_threshold}")
    print(f"Use cross validation: {use_cross_validation}")
    if use_cross_validation:
        print(f"Cross-validation folds: {cv_folds}")
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

                split_definitions = build_trial_splits(
                    trial_records,
                    use_cross_validation=use_cross_validation,
                    cv_folds=cv_folds,
                    random_state=seed,
                )

                for split_def in split_definitions:
                    split_name = split_def["split_name"]
                    fold_id = split_def["fold_id"]
                    fold_suffix = "" if fold_id is None else f"_fold{fold_id:02d}"
                    train_trials = split_def["train_trials"]
                    valid_trials = split_def["valid_trials"]
                    test_trials = split_def["test_trials"]

                    print(f"Running split: {split_name}")
                    print(
                        "Trial split sizes:",
                        len(train_trials),
                        len(valid_trials),
                        len(test_trials),
                    )

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
                        print(
                            "One of the split datasets produced no windows, skipping split."
                        )
                        continue

                    rank_idx, channel_scores, channel_scores_norm = (
                        fisher_score_channels_from_windows_dataset(
                            train_windows_dataset
                        )
                    )
                    n_channels_total = np.array(train_windows_dataset[0][0]).shape[0]
                    top_k_use = min(top_k, n_channels_total)
                    selected_channels = list(rank_idx[:top_k_use])
                    print(
                        f"Total channels: {n_channels_total}, selecting top_k = {top_k_use}"
                    )
                    print("Selected channel indices:", selected_channels)
                    print("Selected channel scores:", channel_scores[selected_channels])
                    print(
                        "Selected normalized channel scores:",
                        channel_scores_norm[selected_channels],
                    )

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
                                    "score_norm": channel_scores_norm[
                                        selected_channels
                                    ],
                                    "split_name": split_name,
                                }
                            ).to_csv(
                                f"{save_dir}/{key}{fold_suffix}_selected_channels.csv",
                                index=False,
                            )
                            print("Saved selected channel info to CSV.")
                    except Exception as e:
                        print("Warning saving channel names:", e)

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

                    X_train = X_train.astype(np.float32)
                    X_valid = X_valid.astype(np.float32)
                    X_test = X_test.astype(np.float32)
                    y_train = y_train.astype(np.int64)
                    y_valid = y_valid.astype(np.int64)
                    y_test = y_test.astype(np.int64)

                    cuda = torch.cuda.is_available()
                    device = "cuda" if cuda else "cpu"
                    if cuda:
                        torch.backends.cudnn.benchmark = False
                    set_random_seeds(seed=seed, cuda=cuda)

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

                    class_weights = compute_class_weight(
                        "balanced", classes=classes, y=y_train
                    )
                    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                        device
                    )
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                    train_sampler, train_class_counts = create_balanced_sampler(y_train)
                    print(f"Training class counts: {train_class_counts}")
                    print("Training sampler: class-balanced WeightedRandomSampler")

                    valid_ds = SkorchDataset(X_valid, y_valid)
                    callbacks = [
                        "accuracy",
                        ("sampled_class_ratio", SampledClassRatioLogger()),
                        (
                            "valid_f1_macro",
                            EpochScoring(
                                scoring="f1_macro",
                                lower_is_better=False,
                                on_train=False,
                                name="valid_f1_macro",
                            ),
                        ),
                        (
                            "valid_balanced_accuracy",
                            EpochScoring(
                                scoring="balanced_accuracy",
                                lower_is_better=False,
                                on_train=False,
                                name="valid_balanced_accuracy",
                            ),
                        ),
                        (
                            "lr_scheduler",
                            LRScheduler(
                                "CosineAnnealingLR", T_max=max(1, n_epochs - 1)
                            ),
                        ),
                    ]
                    if early_stopping_patience > 0:
                        callbacks.append(
                            (
                                "early_stopping",
                                EarlyStopping(
                                    monitor=early_stopping_monitor,
                                    patience=early_stopping_patience,
                                    threshold=early_stopping_threshold,
                                    lower_is_better=early_stopping_lower_is_better,
                                ),
                            )
                        )

                    clf = EEGClassifier(
                        model,
                        criterion=criterion,
                        optimizer=torch.optim.AdamW,
                        train_split=predefined_split(valid_ds),
                        optimizer__lr=lr,
                        optimizer__weight_decay=weight_decay,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        device=device,
                        classes=classes,
                        max_epochs=n_epochs,
                        iterator_train__sampler=train_sampler,
                        iterator_train__shuffle=False,
                    )

                    print("Start training...")
                    clf.fit(X_train, y=y_train)
                    print("Training finished.")
                    summarize_training_history(
                        clf,
                        max_epochs=n_epochs,
                        early_stopping_patience=early_stopping_patience,
                    )

                    y_pred_test = clf.predict(X_test)
                    test_acc = clf.score(X_test, y=y_test)
                    test_f1_macro = f1_score(y_test, y_pred_test, average="macro")
                    test_balanced_accuracy = balanced_accuracy_score(
                        y_test, y_pred_test
                    )
                    print(f"Test acc: {(test_acc * 100):.2f}%")
                    print(f"Test f1_macro: {test_f1_macro:.4f}")
                    print(f"Test balanced_accuracy: {test_balanced_accuracy:.4f}")

                    labels_for_cm = clf.classes_
                    cm_test = confusion_matrix(
                        y_test, y_pred_test, labels=labels_for_cm
                    )

                    resname = f"{key}_"
                    plot_and_save(
                        cm_test,
                        labels_for_cm,
                        f"Confusion Matrix (Test) - {split_name}",
                        f"{save_dir}/{resname}{fold_suffix}_{batch_size}_{n_epochs}_{top_k}_{lr}_cm_test.png",
                    )
                    pd.DataFrame(
                        cm_test, index=labels_for_cm, columns=labels_for_cm
                    ).to_csv(
                        f"{save_dir}/{resname}{fold_suffix}_{batch_size}_{n_epochs}_{top_k}_{lr}_cm_test.csv"
                    )
                    print("Saved results to", save_dir)

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
                        "split_name": split_name,
                        "fold_id": fold_id,
                        "top_k": int(top_k_use),
                        "n_channels_total": int(n_channels_total),
                        "n_channels_selected": int(len(selected_channels)),
                        "n_train": int(X_train.shape[0]),
                        "n_valid": int(X_valid.shape[0]),
                        "n_test": int(X_test.shape[0]),
                        "test_acc": float(test_acc),
                        "test_f1_macro": float(test_f1_macro),
                        "test_balanced_accuracy": float(test_balanced_accuracy),
                        "selected_channel_idx": selected_channels,
                        "selected_channel_scores": [
                            float(s) for s in channel_scores[selected_channels]
                        ],
                        "selected_channel_scores_norm": [
                            float(s) for s in channel_scores_norm[selected_channels]
                        ],
                        "all_channel_scores": [float(s) for s in channel_scores],
                        "all_channel_scores_norm": [
                            float(s) for s in channel_scores_norm
                        ],
                        "selected_channel_names": selected_channel_names,
                    }
                    global_results.append(result_entry)

                    del clf, model
                    del X_train, y_train, X_valid, y_valid, X_test, y_test
                    del train_set, valid_set, test_set
                    del (
                        train_windows_dataset,
                        valid_windows_dataset,
                        test_windows_dataset,
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
    save_cv_summary_files(
        summary_df,
        save_dir,
        f"summary_{n_epochs}_{batch_size}_{top_k}",
    )
    top_k = top_k - TOP_K_STEP
