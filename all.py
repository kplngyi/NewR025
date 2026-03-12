import argparse
import os
import subprocess
import sys
from pathlib import Path


# 统一登记可训练脚本，后面可以用同一套逻辑构造命令，避免为
# fusion / eeg / fnirs 各写一遍重复代码。
SCRIPT_CONFIGS = {
    "fusion": {"file": "fusion.py", "data_dir_arg": "data_dir_fusion"},
    "eeg": {"file": "eeg.py", "data_dir_arg": "data_dir_eeg"},
    "fnirs": {"file": "fnirs.py", "data_dir_arg": "data_dir_fnirs"},
}


def build_base_extra_args(args, script_key):
    # 根据脚本名找到它对应的数据目录参数，并补齐各脚本共用的
    # 运行参数，例如 config、top_k 搜索步长、文件数量限制等。
    config = SCRIPT_CONFIGS[script_key]
    extra_args = [
        "--data_dir",
        getattr(args, config["data_dir_arg"]),
        "--config_path",
        args.config_path,
    ]
    if args.top_k_step is not None:
        extra_args.extend(["--top_k_step", str(args.top_k_step)])
    if args.files_limit is not None:
        extra_args.extend(["--files_limit", str(args.files_limit)])
    if args.min_top_k is not None:
        extra_args.extend(["--min_top_k", str(args.min_top_k)])
    if args.project_root is not None:
        extra_args.extend(["--project_root", args.project_root])
    if args.model is not None:
        extra_args.extend(["--model", args.model])
    if args.early_stop_patience is not None:
        extra_args.extend(["--early_stop_patience", str(args.early_stop_patience)])
    if args.early_stop_monitor is not None:
        extra_args.extend(["--early_stop_monitor", args.early_stop_monitor])
    if args.early_stop_threshold is not None:
        extra_args.extend(["--early_stop_threshold", str(args.early_stop_threshold)])
    return extra_args


def build_command(
    script_path, extra_args, batch_size=None, epochs=None, output_root=None
):
    # 只有在批量扫描 batch_size / epochs 时才显式传参；如果这里
    # 不传，就让子脚本回退到 config.yaml 中的共享默认值。
    cmd = [sys.executable, str(script_path)]
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if output_root is not None:
        cmd.extend(["--output_root", str(output_root)])
    cmd.extend(extra_args)
    return cmd


def run_command(cmd, env=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def start_command(cmd, env=None):
    print("Running:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def get_run_name(batch_size, epochs):
    # 把本轮实验的 batch_size / epochs 编进输出目录名；如果某个值
    # 没在 all.py 里覆盖，就明确标记它来自 config.yaml。
    batch_tag = f"bs{batch_size}" if batch_size is not None else "bs_config"
    epoch_tag = f"ep{epochs}" if epochs is not None else "ep_config"
    return f"{batch_tag}_{epoch_tag}"


def normalize_device_for_child(device):
    # 并行模式下会先通过 CUDA_VISIBLE_DEVICES 把每个子进程限制到
    # 单张卡，因此子脚本内部统一使用 cuda:0 即可。
    if device is None:
        return None
    return "cuda:0" if device.startswith("cuda") else device


def main():
    parser = argparse.ArgumentParser(
        description="Batch launcher for fusion/eeg/fnirs training."
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        choices=list(SCRIPT_CONFIGS.keys()),
        default=["fusion", "eeg", "fnirs"],
        help="Subset of training scripts to run.",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Optional batch-size sweep. If omitted, child scripts use config.yaml.",
    )
    parser.add_argument(
        "--epochs_list",
        type=int,
        nargs="+",
        default=None,
        help="Optional epoch sweep. If omitted, child scripts use config.yaml.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU ids for parallel mode, e.g. --gpus 0 1 2",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run selected scripts in parallel for each parameter combination.",
    )
    # 三个模态的数据目录分开传，是因为各训练脚本读取的数据目录
    # 结构不同，不能共用一个 data_dir。
    parser.add_argument("--data_dir_eeg", type=str, default="PPEEG")
    parser.add_argument("--data_dir_fnirs", type=str, default="PPfNIRS")
    parser.add_argument("--data_dir_fusion", type=str, default="FusionEEG-fNIRS")
    parser.add_argument(
        "--model",
        type=str,
        default="temporal_se",
        choices=["shallow", "temporal_se"],
        help="默认使用 temporal_se，并传给子脚本；也可显式改为 shallow。",
    )
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--project_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="runs")
    parser.add_argument("--files_limit", type=int, default=None)
    parser.add_argument("--top_k_step", type=int, default=None)
    parser.add_argument("--min_top_k", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_monitor", type=str, default=None)
    parser.add_argument("--early_stop_threshold", type=float, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_output_root = Path(args.output_root).expanduser().resolve()
    base_output_root.mkdir(parents=True, exist_ok=True)
    print(f"Script directory: {script_dir}")
    print(f"Batch outputs: {base_output_root}")

    batch_values = args.batch_sizes if args.batch_sizes is not None else [None]
    epoch_values = args.epochs_list if args.epochs_list is not None else [None]

    for batch_size in batch_values:
        for epochs in epoch_values:
            run_name = get_run_name(batch_size, epochs)
            # 每组参数都单独创建输出根目录，避免不同实验的日志、图像、
            # CSV 结果混在一起，后续对比也更方便。
            run_root = base_output_root / run_name

            if args.parallel:
                if args.gpus and len(args.gpus) < len(args.scripts):
                    raise ValueError(
                        f"Need at least {len(args.scripts)} GPU ids for parallel mode; got {len(args.gpus)}"
                    )

                procs = []
                for idx, script_key in enumerate(args.scripts):
                    script_info = SCRIPT_CONFIGS[script_key]
                    script_path = script_dir / script_info["file"]
                    script_output_root = run_root / script_key
                    script_output_root.mkdir(parents=True, exist_ok=True)

                    env = os.environ.copy()
                    extra_args = build_base_extra_args(args, script_key)
                    if args.gpus:
                        # 并行时把每个子进程绑定到单独 GPU，避免多个训练
                        # 任务抢同一张卡。
                        gpu_id = args.gpus[idx]
                        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    if script_key == "eeg" and args.device is not None:
                        eeg_device = (
                            normalize_device_for_child(args.device)
                            if args.gpus
                            else args.device
                        )
                        extra_args.extend(["--device", eeg_device])
                    cmd = build_command(
                        script_path,
                        extra_args,
                        batch_size=batch_size,
                        epochs=epochs,
                        output_root=script_output_root,
                    )
                    procs.append((script_key, start_command(cmd, env=env)))

                for script_key, proc in procs:
                    ret = proc.wait()
                    if ret != 0:
                        raise subprocess.CalledProcessError(ret, script_key)
            else:
                for script_key in args.scripts:
                    script_info = SCRIPT_CONFIGS[script_key]
                    script_path = script_dir / script_info["file"]
                    script_output_root = run_root / script_key
                    script_output_root.mkdir(parents=True, exist_ok=True)

                    extra_args = build_base_extra_args(args, script_key)
                    if script_key == "eeg" and args.device is not None:
                        extra_args.extend(["--device", args.device])
                    cmd = build_command(
                        script_path,
                        extra_args,
                        batch_size=batch_size,
                        epochs=epochs,
                        output_root=script_output_root,
                    )
                    run_command(cmd)


if __name__ == "__main__":
    main()
