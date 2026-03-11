import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import matplotlib
import mne
import numpy as np
import pandas as pd
import torch
import yaml

from braindecode import EEGClassifier
from braindecode.util import set_random_seeds
from fe_u import (
    fisher_score_channels_alpha_beta_from_windows_dataset,
    fisher_score_channels_from_windows_dataset_tdpsd,
)
from mne.datasets import eegbci
from model_factory import build_model
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import Dataset as SkorchDataset
from skorch.helper import predefined_split

from runtime_utils import (
    add_common_runtime_args,
    parse_known_args,
    prepare_runtime_dirs,
    resolve_path,
    resolve_project_root,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVENT_LABELS = {"T0": "Rest", "T1": "Left_Fist", "T2": "Right_Fist"}
EVENT_MAPPING = {"Rest": 0, "Left_Fist": 1, "Right_Fist": 2}
DEFAULT_IMAGERY_RUNS = [4, 8, 12]
DEFAULT_EXECUTION_RUNS = [3, 7, 11]


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
        gpu_index = int(requested.split(":", 1)[1])
        gpu_count = torch.cuda.device_count()
        if gpu_index < 0 or gpu_index >= gpu_count:
            raise RuntimeError(
                f"Requested --device {requested}, but only {gpu_count} CUDA device(s) are available."
            )
        return torch.device(requested)
    raise ValueError(f"Unsupported --device value: {device_arg}")


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


def parse_subjects(subject_values):
    if not subject_values:
        return [1, 2, 3, 4, 5]
    return sorted({int(value) for value in subject_values})


def resolve_runs(task_type, runs_arg):
    if runs_arg:
        return [int(value) for value in runs_arg]
    if task_type == "execution":
        return DEFAULT_EXECUTION_RUNS
    return DEFAULT_IMAGERY_RUNS


def list_existing_edf_files(data_dir):
    if not data_dir.exists():
        return []
    return sorted(data_dir.rglob("*.edf"))


def load_eegbci_paths(subjects, runs, data_dir):
    existing_before = list_existing_edf_files(data_dir)
    print(f"Existing EEGBCI EDF files before load: {len(existing_before)}")
    if existing_before:
        print(f"Using local EEGBCI cache under: {data_dir}")
    else:
        print(f"EEGBCI data not found under {data_dir}, downloading now...")

    paths = eegbci.load_data(subjects, runs, path=str(data_dir), update_path=False)

    existing_after = list_existing_edf_files(data_dir)
    if len(existing_after) > len(existing_before):
        print(
            f"Downloaded {len(existing_after) - len(existing_before)} new EDF file(s) to {data_dir}"
        )
    else:
        print("No new EEGBCI files downloaded; all requested files already existed.")
    return paths


def build_trial_records_from_raw(raw, subject_id, run_id, window_size_samples):
    sfreq = float(raw.info["sfreq"])
    raw_end_time = raw.times[-1] + (1.0 / sfreq)
    annotations = [
        (float(onset), str(desc).strip())
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
        if str(desc).strip() in EVENT_LABELS
    ]

    trial_records = []
    for idx, (onset, desc) in enumerate(annotations):
        next_onset = (
            annotations[idx + 1][0] if idx + 1 < len(annotations) else raw_end_time
        )
        trial_end = min(next_onset, raw_end_time)
        crop_end = max(onset, trial_end - (1.0 / sfreq))
        duration = crop_end - onset + (1.0 / sfreq)
        if duration * sfreq < window_size_samples:
            continue

        label_name = EVENT_LABELS[desc]
        trial_raw = raw.copy().crop(tmin=onset, tmax=crop_end)
        trial_raw.set_annotations(
            mne.Annotations(
                onset=[0.0],
                duration=[duration],
                description=[label_name],
            )
        )
        trial_records.append(
            {
                "trial_id": f"{subject_id}_run{run_id}_{idx}",
                "subject": subject_id,
                "run": int(run_id),
                "label": EVENT_MAPPING[label_name],
                "label_name": label_name,
                "sfreq": sfreq,
                "ch_names": list(trial_raw.ch_names),
                "data": trial_raw.get_data(),
            }
        )
    return trial_records


def stratified_split_trial_records(trial_records, random_state=710):
    if len(trial_records) < 9:
        raise ValueError(f"Need at least 9 trials, got {len(trial_records)}")

    labels = np.array([record["label"] for record in trial_records])
    indices = np.arange(len(trial_records))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=labels,
        random_state=random_state,
    )
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=labels[temp_idx],
        random_state=random_state,
    )
    return (
        [trial_records[i] for i in train_idx],
        [trial_records[i] for i in valid_idx],
        [trial_records[i] for i in test_idx],
    )


def create_window_samples(trial_records, window_size_samples, window_stride_samples):
    samples = []
    for record in trial_records:
        data = np.array(record["data"], dtype=np.float32)
        n_time = data.shape[1]
        if n_time < window_size_samples:
            continue

        starts = list(range(0, n_time - window_size_samples + 1, window_stride_samples))
        last_start = n_time - window_size_samples
        if starts[-1] != last_start:
            starts.append(last_start)

        for start in starts:
            stop = start + window_size_samples
            meta = {
                "subject": record["subject"],
                "run": record["run"],
                "trial_id": record["trial_id"],
                "start_sample": int(start),
                "label_name": record["label_name"],
            }
            samples.append((data[:, start:stop], int(record["label"]), meta))
    return samples


def select_windows_with_channels(samples, selected_channels):
    selected = []
    for X_i, y_i, meta_i in samples:
        X_i_sel = np.array(X_i)[selected_channels, :]
        selected.append((X_i_sel, int(y_i), meta_i))
    return selected


def extract_X_y_from_sample_list(sample_list):
    X_list = [np.array(X, dtype=np.float32) for X, _, _ in sample_list]
    y_list = [int(y) for _, y, _ in sample_list]
    return np.stack(X_list), np.array(y_list, dtype=np.int64)


def label_counts_from_trials(trial_records):
    counts = {name: 0 for name in EVENT_MAPPING}
    for record in trial_records:
        counts[record["label_name"]] += 1
    return counts


def label_counts_from_samples(sample_list):
    inv = {value: key for key, value in EVENT_MAPPING.items()}
    counts = {name: 0 for name in EVENT_MAPPING}
    for _, y_i, _ in sample_list:
        counts[inv[int(y_i)]] += 1
    return counts


def majority_baseline(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


def plot_and_save(cm, labels, title, fname):
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved image: {fname}")


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/eegbci")
parser.add_argument(
    "--task_type", type=str, default="imagery", choices=["imagery", "execution"]
)
parser.add_argument("--subjects", nargs="*", default=[1, 2, 3, 4, 5])
parser.add_argument("--runs", nargs="*", default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--epochs", type=int, default=None)
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
parser.add_argument("--early_stopping_patience", type=int, default=12)
parser.add_argument("--window_size_samples", type=int, default=160)
parser.add_argument("--window_stride_samples", type=int, default=80)
parser.add_argument("--l_freq", type=float, default=1.0)
parser.add_argument("--h_freq", type=float, default=40.0)
args = parse_known_args(add_common_runtime_args(parser))

cwd = os.getcwd()
project_root = resolve_project_root(__file__, args.project_root)
runtime_dirs = prepare_runtime_dirs(project_root, args.output_root)
print("当前工作目录是：", cwd)
print("项目根目录是：", project_root)
os.chdir(project_root)

with open(resolve_path(args.config_path, project_root), "r") as f:
    config = yaml.safe_load(f)

mne.set_log_level("WARNING")

subjects = parse_subjects(args.subjects)
if args.files_limit:
    subjects = subjects[: args.files_limit]
runs = resolve_runs(args.task_type, args.runs)
data_dir = resolve_path(args.data_dir, project_root)
data_dir.mkdir(parents=True, exist_ok=True)

requested_top_k = args.top_k if args.top_k is not None else config.get("top_k", 32)
if requested_top_k <= 0:
    raise ValueError(f"top_k must be a positive integer, got {requested_top_k}")

batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
n_epochs = args.epochs if args.epochs is not None else config["n_epochs"]
lr = config["lr"]
weight_decay = config["weight_decay"]
seed = config["seed"]

now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = runtime_dirs["logs_dir"]
save_dir = runtime_dirs["output_dir"] / "ResEEGBCI"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
out_fname = os.path.join(
    log_dir, f"{now_time}_eegbci_{args.task_type}_{args.model}.log"
)
orig_stdout = sys.stdout
f_out = open(out_fname, "w")
sys.stdout = Tee(orig_stdout, f_out)

try:
    print("\n📌 EEGBCI Quick Test Hyperparameters:")
    print(f"Subjects: {subjects}")
    print(f"Runs: {runs}")
    print(f"Task type: {args.task_type}")
    print(f"Data dir: {data_dir}")
    print(f"窗口大小: {args.window_size_samples} 样本")
    print(f"窗口步长: {args.window_stride_samples} 样本")
    print(f"学习率: {lr}")
    print(f"权重衰减: {weight_decay}")
    print(f"批大小: {batch_size}")
    print(f"训练轮数: {n_epochs}")
    print(f"模型: {args.model}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Fisher方法: {args.fisher_method}")
    print(f"Bandpass: {args.l_freq}-{args.h_freq} Hz")

    eegbci_paths = load_eegbci_paths(subjects, runs, data_dir)
    print(f"Requested EEGBCI file count: {len(eegbci_paths)}")

    trial_records = []
    for path in eegbci_paths:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        eegbci.standardize(raw)
        raw.pick(picks="eeg")
        if args.l_freq is not None or args.h_freq is not None:
            raw.filter(args.l_freq, args.h_freq, fir_design="firwin", verbose="ERROR")

        file_name = Path(path).name
        subject_id = file_name.split("R", 1)[0].replace("S", "")
        run_id = int(file_name.split("R", 1)[1].split(".", 1)[0])
        raw_trials = build_trial_records_from_raw(
            raw,
            subject_id=subject_id,
            run_id=run_id,
            window_size_samples=args.window_size_samples,
        )
        print(f"Loaded {file_name}: retained {len(raw_trials)} labeled segments")
        trial_records.extend(raw_trials)

    if len(trial_records) == 0:
        raise RuntimeError("No EEGBCI trials were extracted.")

    print(f"Total pooled trial count: {len(trial_records)}")
    print("Trial label counts:", label_counts_from_trials(trial_records))

    train_trials, valid_trials, test_trials = stratified_split_trial_records(
        trial_records, random_state=seed
    )
    print(
        "Trial split sizes:",
        len(train_trials),
        len(valid_trials),
        len(test_trials),
    )
    print("Train trial label counts:", label_counts_from_trials(train_trials))
    print("Valid trial label counts:", label_counts_from_trials(valid_trials))
    print("Test trial label counts:", label_counts_from_trials(test_trials))

    train_samples = create_window_samples(
        train_trials,
        window_size_samples=args.window_size_samples,
        window_stride_samples=args.window_stride_samples,
    )
    valid_samples = create_window_samples(
        valid_trials,
        window_size_samples=args.window_size_samples,
        window_stride_samples=args.window_stride_samples,
    )
    test_samples = create_window_samples(
        test_trials,
        window_size_samples=args.window_size_samples,
        window_stride_samples=args.window_stride_samples,
    )

    if min(len(train_samples), len(valid_samples), len(test_samples)) == 0:
        raise RuntimeError("One of the EEGBCI splits produced no windows.")

    print("Train window label counts:", label_counts_from_samples(train_samples))
    print("Valid window label counts:", label_counts_from_samples(valid_samples))
    print("Test window label counts:", label_counts_from_samples(test_samples))
    print(
        f"Valid majority baseline: {majority_baseline(label_counts_from_samples(valid_samples)):.4f}"
    )
    print(
        f"Test majority baseline: {majority_baseline(label_counts_from_samples(test_samples)):.4f}"
    )

    train_sfreq = float(train_trials[0]["sfreq"])
    if args.fisher_method == "bandpower":
        rank_idx, channel_scores = (
            fisher_score_channels_alpha_beta_from_windows_dataset(
                train_samples,
                fs=train_sfreq,
                mode=args.bandpower_mode,
            )
        )
    else:
        rank_idx, channel_scores, _ = fisher_score_channels_from_windows_dataset_tdpsd(
            train_samples
        )

    n_channels_total = np.array(train_samples[0][0]).shape[0]
    top_k_use = min(int(requested_top_k), n_channels_total)
    selected_channels = list(rank_idx[:top_k_use])
    selected_channel_names = [train_trials[0]["ch_names"][i] for i in selected_channels]
    print(f"Total channels: {n_channels_total}, selecting top_k = {top_k_use}")
    print("Selected channel indices:", selected_channels)
    print("Selected channel names:", selected_channel_names)

    train_set = select_windows_with_channels(train_samples, selected_channels)
    valid_set = select_windows_with_channels(valid_samples, selected_channels)
    test_set = select_windows_with_channels(test_samples, selected_channels)

    X_train, y_train = extract_X_y_from_sample_list(train_set)
    X_valid, y_valid = extract_X_y_from_sample_list(valid_set)
    X_test, y_test = extract_X_y_from_sample_list(test_set)

    print(
        "Train/Valid/Test sizes:", X_train.shape[0], X_valid.shape[0], X_test.shape[0]
    )
    print(
        "Shapes (n_windows, n_ch_sel, n_time):",
        X_train.shape,
        X_valid.shape,
        X_test.shape,
    )

    train_device = resolve_training_device(args.device)
    print(f"Requested device: {args.device}")
    print(f"Resolved device: {train_device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if train_device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(train_device)}")

    set_random_seeds(seed=seed, cuda=train_device.type == "cuda")
    classes = np.unique(y_train)

    model = build_model(
        args.model,
        n_channels=X_train.shape[1],
        n_classes=len(classes),
        n_times=X_train.shape[2],
    )
    if train_device.type == "cuda":
        model.cuda()
    print("Model:", args.model)
    print(model)

    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = torch.FloatTensor(class_weights).to(train_device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    valid_ds = SkorchDataset(X_valid, y_valid)
    callbacks = [
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, n_epochs - 1))),
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

    clf = EEGClassifier(
        model,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_ds),
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

    y_pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    test_macro_f1 = f1_score(y_test, y_pred_test, average="macro")
    print(f"Test acc: {(test_acc * 100):.2f}%")
    print(f"Test balanced_acc: {test_bal_acc:.4f}")
    print(f"Test macro_f1: {test_macro_f1:.4f}")

    labels_for_cm = ["Rest", "Left_Fist", "Right_Fist"]
    cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])
    cm_png = os.path.join(
        save_dir, f"{now_time}_{args.task_type}_{args.model}_cm_test.png"
    )
    cm_csv = os.path.join(
        save_dir, f"{now_time}_{args.task_type}_{args.model}_cm_test.csv"
    )
    plot_and_save(cm_test, labels_for_cm, "EEGBCI Confusion Matrix (Test)", cm_png)
    pd.DataFrame(cm_test, index=labels_for_cm, columns=labels_for_cm).to_csv(cm_csv)

    selected_channels_csv = os.path.join(
        save_dir, f"{now_time}_{args.task_type}_{args.model}_selected_channels.csv"
    )
    pd.DataFrame(
        {
            "selected_channel_idx": selected_channels,
            "selected_channel_name": selected_channel_names,
            "score": channel_scores[selected_channels],
        }
    ).to_csv(selected_channels_csv, index=False)

    summary_entry = {
        "timestamp": now_time,
        "task_type": args.task_type,
        "subjects": json.dumps(subjects),
        "runs": json.dumps(runs),
        "model": args.model,
        "fisher_method": args.fisher_method,
        "top_k": int(top_k_use),
        "window_size_samples": int(args.window_size_samples),
        "window_stride_samples": int(args.window_stride_samples),
        "n_trials_total": int(len(trial_records)),
        "n_train_trials": int(len(train_trials)),
        "n_valid_trials": int(len(valid_trials)),
        "n_test_trials": int(len(test_trials)),
        "n_train_windows": int(X_train.shape[0]),
        "n_valid_windows": int(X_valid.shape[0]),
        "n_test_windows": int(X_test.shape[0]),
        "valid_majority_baseline": float(
            majority_baseline(label_counts_from_samples(valid_samples))
        ),
        "test_majority_baseline": float(
            majority_baseline(label_counts_from_samples(test_samples))
        ),
        "test_acc": float(test_acc),
        "test_balanced_acc": float(test_bal_acc),
        "test_macro_f1": float(test_macro_f1),
        "selected_channel_idx": json.dumps([int(x) for x in selected_channels]),
        "selected_channel_names": json.dumps(selected_channel_names),
    }
    summary_csv = os.path.join(
        save_dir, f"{now_time}_{args.task_type}_{args.model}_summary.csv"
    )
    pd.DataFrame([summary_entry]).to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")
finally:
    sys.stdout = orig_stdout
    f_out.close()
