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
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--data_dir', type=str, default='PPEEG')
parser.add_argument('--device', type=str, default='auto')
parser.add_argument('--model', type=str, default='temporal_se', choices=['shallow', 'temporal_se'])
args = parse_known_args(add_common_runtime_args(parser))

# 切换到项目根目录，兼容本地脚本和 Colab 工作目录
cwd = os.getcwd()
project_root = resolve_project_root(__file__, args.project_root)
runtime_dirs = prepare_runtime_dirs(project_root, args.output_root)
print("当前工作目录是：", cwd)
print("项目根目录是：", project_root)
os.chdir(project_root)


# 提取fnirs 的fif文件

import os
import mne
import numpy as np
target_path = resolve_path(args.data_dir, project_root)
filesA1 = [str(target_path / f) for f in os.listdir(target_path) if f.endswith('.fif') and 'A1' in f and not f.startswith('.')]
filesA1 = sorted(filesA1)
if args.files_limit:
    filesA1 = filesA1[:args.files_limit]
print(filesA1)

# filesA1 = filesA1[-2:]
# full_pipeline_channel_selection_then_train.py
import os
import sys
import gc
import datetime

import numpy as np
import pandas as pd
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

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split


# 超参数
# from config import config
import yaml
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
            raise RuntimeError(f"Requested --device {requested} but CUDA is not available.")
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

# # ------------------ 输出重定向（可选） ------------------
# now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# sys.stdout = open(f'{now_time}_traineeg.md', 'w')


# ------------------ 通道选择（Fisher score） ------------------
def fisher_score_channels_from_windows_dataset(windows_dataset):
    """
    用每个窗口每个通道的 mean(abs(x)) 作为通道特征，按 Fisher score 排序通道。
    返回 rank_idx（按分数从大到小的通道索引数组）和 scores（每通道分数）。
    """
    n_windows = len(windows_dataset)
    # 获取通道数
    sample_X, sample_y, _ = windows_dataset[0]
    n_channels = np.array(sample_X).shape[0]
    F = np.zeros((n_windows, n_channels))
    ys = np.zeros((n_windows,), dtype=int)
    for i in range(n_windows):
        X_i, y_i, _ = windows_dataset[i]
        X_i = np.array(X_i)
        # 特征：mean(abs(.)) over time
        F[i, :] = np.mean(np.abs(X_i), axis=1)
        ys[i] = int(y_i)
    # 计算 Fisher score
    classes = np.unique(ys)
    mu_total = F.mean(axis=0)
    Sb = np.zeros(n_channels)
    Sw = np.zeros(n_channels)
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
top_k = MAX_CHANNELS
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
    now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = runtime_dirs["logs_dir"]
    save_dir = runtime_dirs["output_dir"] / "ResEEG"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    out_fname = os.path.join(log_dir, f'{now_time}_{top_k}_{n_epochs}_{lr}_traineeg.log')
    orig_stdout = sys.stdout
    f_out = open(out_fname, 'w')
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
    try:
        print("Files to process:", filesA1)
        if len(filesA1) == 0:
            print("No .fif files found in target_path. Exiting.")
        # ------------------ 处理每个 fif 文件循环 ------------------
        # 你需要在外部定义 filesA1 = [...] 列表（fif 文件路径）

        for filea1 in filesA1:
            resname = os.path.basename(filea1[:-4])
            print("Processing file:", resname)
            subject_id = resname[:4]
            print("subject_id:", subject_id)

            # 读取 raw（copy 避免原始对象被修改）
            raw = mne.io.read_raw_fif(filea1, preload=True)
            raw = raw.copy()

            # 通过注释创建事件（你原有逻辑）
            events, event_id = mne.events_from_annotations(raw)
            print("Original event_id:", event_id)
            # 你的文件中目标注释描述
            target_desc = 'Stimulus/S  5'
            annotations = raw.annotations
            matched_onsets = [onset for onset, desc in zip(annotations.onset, annotations.description)
                            if desc.strip() == target_desc.strip()]
            print("Matched onsets for 'Stimulus/S  5':", matched_onsets)
            if len(matched_onsets) == 0:
                print("No matched onsets found, skipping file.")
                continue
            t0 = matched_onsets[0]
            n_trials = len(matched_onsets)
            print(f"n_trials: {n_trials}")

            # 构建新的注释段（Rest, Elbow_Flexion, Elbow_Extension）
            onsets = []
            durations = []
            descriptions = []
            rest_time = 1e10
            ext_time = 1e10
            for flex_time in matched_onsets:
                onsets.append(min(ext_time+ext_dur, flex_time - rest_dur))
                durations.append(rest_dur)
                descriptions.append('Rest')

                onsets.append(flex_time)
                durations.append(flex_dur)
                descriptions.append('Elbow_Flexion')

                ext_time = flex_time + flex_dur
                onsets.append(ext_time)
                durations.append(ext_dur)
                descriptions.append('Elbow_Extension')

            new_annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=raw.annotations.orig_time)
            raw.set_annotations(raw.annotations + new_annotations)

            # 自定义事件 id
            custom_event_id = {'Rest': 0, 'Elbow_Flexion': 1, 'Elbow_Extension': 2}
            events, event_id = mne.events_from_annotations(raw, event_id=custom_event_id)
            print("Custom event_id:", event_id)

            # 使用 braindecode create_from_mne_raw 生成 windows_dataset
            descriptions_braindecode = [{"event_code": [0, 1, 2], "subject": subject_id}]
            mapping = {'Rest': 0, 'Elbow_Flexion': 1, 'Elbow_Extension': 2}
            parts = [raw]

            windows_dataset = create_from_mne_raw(
                    parts,
                    trial_start_offset_samples=0,
                    trial_stop_offset_samples=0,
                    window_size_samples=window_size_samples,
                    window_stride_samples=window_stride_samples,
                    drop_last_window=False,
                    descriptions=descriptions_braindecode,
                    mapping=mapping,
                )
            print("Created windows_dataset, length:", len(windows_dataset))
            

            rank_idx, channel_scores = fisher_score_channels_from_windows_dataset(windows_dataset)
            n_channels_total = np.array(windows_dataset[0][0]).shape[0]
            top_k_use = min(top_k, n_channels_total)
            selected_channels = list(rank_idx[:top_k_use])
            print(f"Total channels: {n_channels_total}, selecting top_k = {top_k_use}")
            print("Selected channel indices:", selected_channels)
            print("Selected channel scores:", channel_scores[selected_channels])

            # ------------------ 用选定通道构建样本列表 (X_sel, y, meta) ------------------
            all_samples = []
            for i in range(len(windows_dataset)):
                X_i, y_i, meta_i = windows_dataset[i]
                X_i = np.array(X_i)  # (n_ch, n_time)
                X_i_sel = X_i[selected_channels, :]
                all_samples.append((X_i_sel, int(y_i), meta_i))

            # 提取标签并划分 train/valid/test（70/15/15，stratify）
            labels = np.array([s[1] for s in all_samples])
            if len(np.unique(labels)) < 2:
                print("Less than 2 classes found, skipping file.")
                continue

            train_idx, temp_idx = train_test_split(
                range(len(all_samples)),
                test_size=0.3,
                stratify=labels,
                random_state=710
            )
            valid_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.5,
                stratify=labels[temp_idx],
                random_state=710
            )
            train_set = [all_samples[i] for i in train_idx]
            valid_set = [all_samples[i] for i in valid_idx]
            test_set  = [all_samples[i] for i in test_idx]

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
            X_test,  y_test  = extract_X_y_from_sample_list(test_set)

            print("Train/Valid/Test sizes:", X_train.shape[0], X_valid.shape[0], X_test.shape[0])
            print("Shapes (n_trials, n_ch_sel, n_time):", X_train.shape, X_valid.shape, X_test.shape)
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
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = torch.FloatTensor(class_weights).to(train_device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

            # ------------------ 构建 EEGClassifier 并训练 ------------------
            clf = EEGClassifier(
                model,
                criterion=criterion,
                optimizer=torch.optim.AdamW,
                train_split=predefined_split(valid_set),
                optimizer__lr=lr,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                callbacks=[
                    "accuracy",
                    ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, n_epochs - 1))),
                ],
                device=str(train_device),
                classes=classes,
                max_epochs=n_epochs,
            )

            print("Start training...")
            clf.fit(X_train, y=y_train)
            print("Training finished.")

            # ------------------ 评估 ------------------
            y_pred_test = clf.predict(X_test)
            test_acc = clf.score(X_test, y=y_test)
            print(f"Test acc: {(test_acc * 100):.2f}%")

            labels_for_cm = clf.classes_
            cm_test = confusion_matrix(y_test, y_pred_test, labels=labels_for_cm)

            # ---------- 保存结果 ----------
            def plot_and_save(cm, labels, title, fname):
                disp = ConfusionMatrixDisplay(cm, display_labels=labels)
                fig, ax = plt.subplots(figsize=(4,4))
                disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format='d')
                ax.set_title(title)
                fig.tight_layout()
                fig.savefig(fname, dpi=300)
                plt.close(fig)
                print(f"Saved image: {fname}")

            plot_and_save(cm_test, labels_for_cm, "Confusion Matrix (Test)", os.path.join(save_dir, f"{resname}_{batch_size}_{n_epochs}_{top_k}_{lr}_cm_test.png"))
            pd.DataFrame(cm_test, index=labels_for_cm, columns=labels_for_cm).to_csv(os.path.join(save_dir, f"{resname}_{batch_size}_{n_epochs}_{top_k}_{lr}_cm_test.csv"))
            print("CSV 已保存到", save_dir)

            # 记录所选通道到文件，便于审查
            pd.DataFrame({
                "selected_channel_idx": selected_channels,
                "score": channel_scores[selected_channels]
            }).to_csv(os.path.join(save_dir, f"{resname}_{top_k}_selected_channels.csv"), index=False)
            print("Selected channels saved.")

            # ---------- 将结果记录到 global_results（用于后续比较） ----------
            try:
                ch_names = raw.info.get('ch_names', None)
                selected_channel_names = [ch_names[i] for i in selected_channels] if ch_names is not None else []
            except Exception:
                selected_channel_names = []

            result_entry = {
                'subject': subject_id,
                'file': resname,
                'top_k': int(top_k_use),
                'n_channels_total': int(n_channels_total),
                'n_channels_selected': int(len(selected_channels)),
                'n_train': int(X_train.shape[0]),
                'n_valid': int(X_valid.shape[0]),
                'n_test': int(X_test.shape[0]),
                'test_acc': float(test_acc),
                'selected_channel_idx': selected_channels,
                'selected_channel_names': selected_channel_names,
            }
            global_results.append(result_entry)

            # ------------------ 清理释放内存 / GPU ------------------
            del clf, model
            del X_train, y_train, X_valid, y_valid, X_test, y_test
            del train_set, valid_set, test_set, all_samples, windows_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            gc.collect()

        # 关闭重定向（如果需要在脚本结束后恢复输出）
            # 恢复 stdout 并关闭文件
    finally:
        sys.stdout = orig_stdout
        f_out.close()
        print(f"Script finished. Log saved to {out_fname}")
    ## 保存csv
    summary_df = pd.DataFrame(global_results)
    summary_csv = os.path.join(save_dir, f"summary_{n_epochs}_{batch_size}_{top_k}_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("Saved summary CSV:", summary_csv)
    top_k = top_k - TOP_K_STEP
