import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_script(script_path, batch_size, epochs, extra_args):
    cmd = [
        sys.executable,
        str(script_path),
        "--batch_size",
        str(batch_size),
        "--epochs",
        str(epochs),
    ]
    cmd.extend(extra_args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def start_script(script_path, batch_size, epochs, extra_args, env):
    cmd = [
        sys.executable,
        str(script_path),
        "--batch_size",
        str(batch_size),
        "--epochs",
        str(epochs),
    ]
    cmd.extend(extra_args)
    print("Running:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Train fusion/eeg/fnirs pipelines in sequence.")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--epochs_list", type=int, nargs="+", default=[30,50,100])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir_eeg", type=str, default="PPEEG")
    parser.add_argument("--data_dir_fnirs", type=str, default="PPfNIRS")
    parser.add_argument("--data_dir_fusion", type=str, default="FusionEEG-fNIRS")
    parser.add_argument('--model', type=str, default='temporal_se', choices=['shallow', 'temporal_se'])
    parser.add_argument("--files_limit", type=int, default=None)
    parser.add_argument("--parallel", action="store_true", help="Run fusion/eeg/fnirs in parallel.")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU ids for parallel mode, e.g. --gpus 0 1 2")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    print(f"Script directory: {script_dir}")
    train_order = [
        ("fusion.py", args.data_dir_fusion),
        ("eeg.py", args.data_dir_eeg),
        ("fnirs.py", args.data_dir_fnirs),
    ]

    for batch in args.batch_sizes:
        for ep in args.epochs_list:
            if args.parallel:
                if not args.gpus:
                    raise ValueError("--parallel requires --gpus, e.g. --gpus 0 1 2")
                if len(args.gpus) < len(train_order):
                    raise ValueError(
                        f"Need at least {len(train_order)} GPU ids for parallel mode; got {len(args.gpus)}"
                    )

                procs = []
                for idx, (script_name, data_dir) in enumerate(train_order):
                    script_path = script_dir / script_name
                    gpu_id = args.gpus[idx]
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                    # When only one GPU is visible, child scripts should use cuda:0.
                    child_device = "cuda:0" if args.device.startswith("cuda") else args.device
                    extra_args = [
                        "--device", child_device,
                        "--data_dir", data_dir,
                        "--model", args.model,
                    ]
                    if args.files_limit is not None:
                        extra_args.extend(["--files_limit", str(args.files_limit)])

                    p = start_script(script_path, batch, ep, extra_args, env)
                    procs.append((script_name, gpu_id, p))

                for script_name, gpu_id, p in procs:
                    ret = p.wait()
                    if ret != 0:
                        raise subprocess.CalledProcessError(ret, f"{script_name} on GPU {gpu_id}")
            else:
                for script_name, data_dir in train_order:
                    script_path = script_dir / script_name
                    extra_args = [
                        "--device", args.device,
                        "--data_dir", data_dir,
                        "--model", args.model,
                    ]
                    if args.files_limit is not None:
                        extra_args.extend(["--files_limit", str(args.files_limit)])
                    run_script(script_path, batch, ep, extra_args)


if __name__ == "__main__":
    main()