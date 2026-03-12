import os
import mne
import numpy as np
import pandas as pd
import math

import argparse
from runtime_utils import (
    add_common_runtime_args,
    parse_known_args,
    prepare_runtime_dirs,
    resolve_path,
    resolve_project_root,
)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--early_stop_patience", type=int, default=None)
parser.add_argument("--early_stop_monitor", type=str, default=None)
parser.add_argument("--early_stop_threshold", type=float, default=None)
parser.add_argument("--data_dir", type=str, default="PPfNIRS")
parser.add_argument(
    "--model", type=str, default="shallow", choices=["shallow", "temporal_se"]
)
args = parse_known_args(add_common_runtime_args(parser))

cwd = os.getcwd()
project_root = resolve_project_root(__file__, args.project_root)
runtime_dirs = prepare_runtime_dirs(project_root, args.output_root)
target_path = resolve_path(args.data_dir, project_root)

print("当前工作目录是：", cwd)
print("项目根目录是：", project_root)
os.chdir(project_root)
filesA1 = [
    str(target_path / f)
    for f in os.listdir(target_path)
    if f.endswith(".fif") and "A1" in f
]
filesA1 = sorted(filesA1)
if args.files_limit:
    filesA1 = filesA1[: args.files_limit]
print(filesA1)

import os
import sys
import gc
import datetime
import traceback

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import mne

from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from braindecode.datasets import create_from_mne_raw
from braindecode import EEGClassifier
from braindecode.util import set_random_seeds
from model_factory import build_model

from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split
from skorch.dataset import Dataset as SkorchDataset

from fnirs_pair_fisher import (
    compute_pair_fisher_scores,
    expand_pair_selection_to_channels,
)

import yaml

with open(resolve_path(args.config_path, project_root), "r") as f:
    config = yaml.safe_load(f)
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


# 这里把 top_k 的含义改成“位置对数”，而不是 88 个独立通道数。
# 每选中 1 个位置，会同时保留该位置对应的 HbO 和 HbR 两个通道。
MAX_CHANNEL_PAIRS = 44
MIN_TOP_K = math.ceil(MAX_CHANNEL_PAIRS * 0.1)
if args.min_top_k is not None:
    MIN_TOP_K = args.min_top_k
TOP_K_STEP = (
    args.top_k_step if args.top_k_step is not None else config.get("top_k_step", 10)
)
top_k = MAX_CHANNEL_PAIRS
window_size_samples = config["window_size_samples"]
window_stride_samples = config["window_stride_samples"]
batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
n_epochs = args.epochs if args.epochs is not None else config["n_epochs"]
early_stop_patience = max(
    1,
    args.early_stop_patience
    if args.early_stop_patience is not None
    else config.get("early_stop_patience", 20),
)
early_stop_monitor = (
    args.early_stop_monitor
    if args.early_stop_monitor is not None
    else config.get("early_stop_monitor", "valid_loss")
)
early_stop_threshold = (
    args.early_stop_threshold
    if args.early_stop_threshold is not None
    else config.get("early_stop_threshold", 1e-3)
)
lr = config["lr"]
weight_decay = config["weight_decay"]
seed = config["seed"]
PAIR_FEATURE_DESC = "mean-std-slope"
PAIR_WEIGHT_HBO = 0.7
PAIR_WEIGHT_HBR = 0.3
PAIR_NORM_METHOD = "minmax"
PAIR_RESULT_LABEL = f"fnirs_pair_norm-{PAIR_NORM_METHOD}_feat-{PAIR_FEATURE_DESC}_hbo{PAIR_WEIGHT_HBO}-hbr{PAIR_WEIGHT_HBR}"

save_dir = runtime_dirs["output_dir"] / "ResfNIRS_pair"
os.makedirs(save_dir, exist_ok=True)


def extract_X_y_from_sample_list(sample_list):
    X_list = []
    y_list = []
    for X, y, _ in sample_list:
        X_list.append(np.array(X))
        y_list.append(int(y))
    X_all = np.stack(X_list)
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


while top_k >= MIN_TOP_K:
    global_results = []
    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = runtime_dirs["logs_dir"]
    os.makedirs(log_dir, exist_ok=True)
    out_fname = os.path.join(
        log_dir, f"{now_time}_{top_k}_{n_epochs}_{lr}_trainfnirs_pair.log"
    )
    orig_stdout = sys.stdout
    f_out = open(out_fname, "w")
    sys.stdout = Tee(orig_stdout, f_out)

    print("\n📌 Training Hyperparameters:")
    print(f"保留的位置对数: {top_k}")
    print(f"窗口大小: {window_size_samples} 样本")
    print(f"窗口步长: {window_stride_samples} 样本")
    print(f"学习率: {lr}")
    print(f"权重衰减: {weight_decay}")
    print(f"批大小: {batch_size}")
    print(f"训练轮数: {n_epochs}")
    print(f"早停监控指标: {early_stop_monitor}")
    print(f"早停耐心轮数: {early_stop_patience}")
    print(f"早停最小改进幅度: {early_stop_threshold}")
    print(f"模态最大位置对数: {MAX_CHANNEL_PAIRS}")
    print(f"最小搜索位置对数: {MIN_TOP_K}")
    print(f"位置对搜索步长: {TOP_K_STEP}")

    try:
        print("Files to process:", filesA1)
        if len(filesA1) == 0:
            print("No .fif files found in target_path. Exiting.")
        for file in filesA1:
            model = None
            clf = None
            try:
                resname = os.path.basename(file[:-4])
                print("\n--- Processing file:", resname, " ---")
                subject_id = resname[:4]
                print("subject_id:", subject_id)

                raw = mne.io.read_raw_fif(file, preload=True)
                raw = raw.copy()

                print(
                    "Sample annotations (first 10):",
                    list(raw.annotations.description[:10]),
                )
                events, event_id = mne.events_from_annotations(raw)
                print("Original event_id:", event_id)

                target_desc = "11"
                annotations = raw.annotations
                matched_onsets = [
                    onset
                    for onset, desc in zip(annotations.onset, annotations.description)
                    if desc.strip() == target_desc.strip()
                ]
                print(
                    "Matched onsets for target_desc '{}': {}".format(
                        target_desc, matched_onsets
                    )
                )
                if len(matched_onsets) == 0:
                    print("No matched onsets found for this file. Skipping.")
                    continue

                rest_dur = 20
                flex_dur = 10
                ext_dur = 5

                onsets = []
                durations = []
                descriptions = []
                ext_time = 1e10
                for flex_time in matched_onsets:
                    rest_time = min(ext_time + ext_dur, flex_time - rest_dur)
                    onsets.append(rest_time)
                    durations.append(rest_dur)
                    descriptions.append("Rest")
                    onsets.append(flex_time)
                    durations.append(flex_dur)
                    descriptions.append("Elbow_Flexion")
                    ext_time = flex_time + flex_dur
                    onsets.append(ext_time)
                    durations.append(ext_dur)
                    descriptions.append("Elbow_Extension")

                new_annotations = mne.Annotations(
                    onset=onsets,
                    duration=durations,
                    description=descriptions,
                    orig_time=raw.annotations.orig_time,
                )
                raw.set_annotations(raw.annotations + new_annotations)

                custom_event_id = {"Rest": 0, "Elbow_Flexion": 1, "Elbow_Extension": 2}
                events, event_id = mne.events_from_annotations(
                    raw, event_id=custom_event_id
                )
                print("Custom event_id:", event_id)

                parts = [raw]
                descriptions_bd = [{"event_code": [0, 1, 2], "subject": subject_id}]
                mapping = {"Rest": 0, "Elbow_Flexion": 1, "Elbow_Extension": 2}

                windows_dataset = create_from_mne_raw(
                    parts,
                    trial_start_offset_samples=0,
                    trial_stop_offset_samples=0,
                    window_size_samples=window_size_samples,
                    window_stride_samples=window_stride_samples,
                    drop_last_window=False,
                    descriptions=descriptions_bd,
                    mapping=mapping,
                )

                print("Created windows_dataset, length:", len(windows_dataset))
                if len(windows_dataset) == 0:
                    print("No windows created. Skipping file.")
                    continue

                ch_names = raw.info.get("ch_names", None)
                if ch_names is None:
                    raise ValueError(
                        "raw.info 中没有 ch_names，无法进行 HbO/HbR 配对。"
                    )

                pair_rank_idx, pair_scores, pair_infos, pair_detail_rows = (
                    compute_pair_fisher_scores(windows_dataset, ch_names)
                )
                n_pairs_total = len(pair_infos)
                top_k_use = min(top_k, n_pairs_total)
                selected_pair_indices = list(pair_rank_idx[:top_k_use])
                selected_channels, selected_pair_rows = (
                    expand_pair_selection_to_channels(selected_pair_indices, pair_infos)
                )
                selected_pair_scores = [pair_scores[i] for i in selected_pair_indices]

                print(
                    f"Total pair count: {n_pairs_total}, selecting top_k pairs = {top_k_use}"
                )
                print(
                    f"Expanded channel count used for training: {len(selected_channels)}"
                )
                print("Selected pair indices:", selected_pair_indices)
                print(
                    "Selected pair names:",
                    [pair_infos[i]["pair_name"] for i in selected_pair_indices],
                )
                print("Selected pair scores:", selected_pair_scores)
                print("Expanded channel indices:", selected_channels)

                pair_detail_df = pd.DataFrame(pair_detail_rows)
                pair_detail_df.to_csv(
                    f"{save_dir}/{resname}_{PAIR_RESULT_LABEL}_pair_scores.csv",
                    index=False,
                )

                selected_pair_export_rows = []
                for pair_idx in selected_pair_indices:
                    pair_info = pair_infos[pair_idx]
                    pair_row = pair_detail_df.iloc[pair_idx].to_dict()
                    pair_row.update(
                        {
                            "selected": True,
                            "expanded_channel_indices": f"{pair_info['hbo_idx']},{pair_info['hbr_idx']}",
                        }
                    )
                    selected_pair_export_rows.append(pair_row)
                pd.DataFrame(selected_pair_export_rows).to_csv(
                    f"{save_dir}/{resname}_{PAIR_RESULT_LABEL}_selected_pairs.csv",
                    index=False,
                )
                print("Saved selected pair info to CSV.")

                selected_windows = []
                for i in range(len(windows_dataset)):
                    X_i, y_i, meta_i = windows_dataset[i]
                    X_i = np.array(X_i)
                    X_i_sel = X_i[selected_channels, :]
                    selected_windows.append((X_i_sel, int(y_i), meta_i))

                labels = np.array([s[1] for s in selected_windows])
                if len(np.unique(labels)) < 2:
                    print("Less than 2 classes found after selection, skipping file.")
                    continue

                indices = np.arange(len(selected_windows))
                train_idx, temp_idx = train_test_split(
                    indices, test_size=0.3, stratify=labels, random_state=710
                )
                valid_idx, test_idx = train_test_split(
                    temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=710
                )

                train_set = [selected_windows[i] for i in train_idx]
                valid_set = [selected_windows[i] for i in valid_idx]
                test_set = [selected_windows[i] for i in test_idx]

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
                class_weights = torch.FloatTensor(class_weights).to(device)
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

                valid_ds = SkorchDataset(X_valid, y_valid)

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
                        (
                            "early_stopping",
                            EarlyStopping(
                                monitor=early_stop_monitor,
                                patience=early_stop_patience,
                                threshold=early_stop_threshold,
                                threshold_mode="abs",
                                lower_is_better=(early_stop_monitor == "valid_loss"),
                                load_best=True,
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
                actual_epochs = len(clf.history)
                early_stopping_cb = None
                callbacks_runtime = getattr(clf, "callbacks_", None)
                if isinstance(callbacks_runtime, dict):
                    early_stopping_cb = callbacks_runtime.get("early_stopping")
                elif callbacks_runtime is not None:
                    for callback_name, callback_obj in callbacks_runtime:
                        if callback_name == "early_stopping":
                            early_stopping_cb = callback_obj
                            break
                best_valid_score = getattr(early_stopping_cb, "best_score_", None)
                best_epoch = getattr(early_stopping_cb, "best_epoch_", None)
                print(f"Actual trained epochs: {actual_epochs}")
                print(f"Best {early_stop_monitor}: {best_valid_score}")
                print(f"Best epoch: {best_epoch}")

                y_pred_test = clf.predict(X_test)
                test_acc = clf.score(X_test, y=y_test)
                print(f"Test acc: {(test_acc * 100):.2f}%")

                labels_for_cm = clf.classes_
                cm_test = confusion_matrix(y_test, y_pred_test, labels=labels_for_cm)

                plot_and_save(
                    cm_test,
                    labels_for_cm,
                    "Confusion Matrix (Test)",
                    f"{save_dir}/{resname}_{batch_size}_{n_epochs}_{top_k}_{lr}_cm_test.png",
                )
                pd.DataFrame(
                    cm_test, index=labels_for_cm, columns=labels_for_cm
                ).to_csv(
                    f"{save_dir}/{resname}_{batch_size}_{n_epochs}_{top_k}_{lr}_cm_test.csv",
                    index=True,
                )
                print("CSV 已保存到", save_dir)

                selected_pair_names = [
                    pair_infos[i]["pair_name"] for i in selected_pair_indices
                ]
                selected_channel_names = [ch_names[i] for i in selected_channels]
                pair_score_mean = float(pair_scores.mean())
                pair_score_std = float(pair_scores.std())
                pair_score_min = float(pair_scores.min())
                pair_score_max = float(pair_scores.max())

                result_entry = {
                    "subject": subject_id,
                    "file": resname,
                    "top_k_pairs": int(top_k_use),
                    "n_pairs_total": int(n_pairs_total),
                    "n_channels_selected": int(len(selected_channels)),
                    "n_train": int(X_train.shape[0]),
                    "n_valid": int(X_valid.shape[0]),
                    "n_test": int(X_test.shape[0]),
                    "trained_epochs": int(actual_epochs),
                    "early_stop_monitor": early_stop_monitor,
                    "early_stop_patience": int(early_stop_patience),
                    "best_valid_score": None
                    if best_valid_score is None
                    else float(best_valid_score),
                    "best_epoch": None if best_epoch is None else int(best_epoch),
                    "test_acc": float(test_acc),
                    "selected_pair_idx": selected_pair_indices,
                    "selected_pair_names": selected_pair_names,
                    "selected_channel_idx": selected_channels,
                    "selected_channel_names": selected_channel_names,
                    "selected_pair_scores_norm": [
                        float(pair_scores[i]) for i in selected_pair_indices
                    ],
                    "score_norm_scope": "global_all_channels",
                    "pair_score_mean": pair_score_mean,
                    "pair_score_std": pair_score_std,
                    "pair_score_min": pair_score_min,
                    "pair_score_max": pair_score_max,
                    "fisher_norm_method": PAIR_NORM_METHOD,
                    "fisher_feature_desc": PAIR_FEATURE_DESC,
                }
                global_results.append(result_entry)

                del clf, model
                del X_train, y_train, X_valid, y_valid, X_test, y_test
                del train_set, valid_set, test_set, selected_windows, windows_dataset
                del pair_rank_idx, pair_scores, pair_infos, selected_channels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                gc.collect()

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                traceback.print_exc()
                try:
                    if model is not None:
                        del model
                    if clf is not None:
                        del clf
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    pass
                continue
    finally:
        sys.stdout = orig_stdout
        f_out.close()
        print(f"Script finished. Log saved to {out_fname}")

    summary_df = pd.DataFrame(global_results)
    summary_csv = os.path.join(
        save_dir,
        f"summary_{PAIR_RESULT_LABEL}_ep{n_epochs}_bs{batch_size}_topk{top_k}_step{TOP_K_STEP}_{now_time}.csv",
    )
    summary_df.to_csv(summary_csv, index=False)
    print("Saved summary CSV:", summary_csv)
    top_k = top_k - TOP_K_STEP
