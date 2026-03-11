import argparse
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


def run_group_in_parallel(tasks):
    """Launch a group of commands in parallel and wait for all to finish."""
    procs = []
    for task in tasks:
        cmd = task["cmd"]
        print("Running (parallel):", " ".join(cmd))
        procs.append((task["name"], subprocess.Popen(cmd)))

    failures = []
    for name, proc in procs:
        ret = proc.wait()
        if ret != 0:
            failures.append((name, ret))

    if failures:
        lines = [f"{name} exited with code {code}" for name, code in failures]
        raise RuntimeError("Parallel run failed: " + "; ".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Train fusion/eeg/fnirs pipelines in sequence.")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--epochs_list", type=int, nargs="+", default=[30,50,100])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="Optional device list used for parallel mode, e.g. cuda:0 cuda:1 cuda:2",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run fusion/eeg/fnirs in parallel for each (batch, epoch) pair.",
    )
    parser.add_argument("--data_dir_eeg", type=str, default="PPEEG")
    parser.add_argument("--data_dir_fnirs", type=str, default="PPfNIRS")
    parser.add_argument("--data_dir_fusion", type=str, default="FusionEEG-fNIRS")
    parser.add_argument('--model', type=str, default='temporal_se', choices=['shallow', 'temporal_se'])
    parser.add_argument("--files_limit", type=int, default=None)
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
                devices = args.devices if args.devices else [args.device]
                tasks = []
                for idx, (script_name, data_dir) in enumerate(train_order):
                    script_path = script_dir / script_name
                    assigned_device = devices[idx % len(devices)]
                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--batch_size",
                        str(batch),
                        "--epochs",
                        str(ep),
                        "--device",
                        assigned_device,
                        "--data_dir",
                        data_dir,
                        "--model",
                        args.model,
                    ]
                    if args.files_limit is not None:
                        cmd.extend(["--files_limit", str(args.files_limit)])
                    tasks.append({"name": script_name, "cmd": cmd})
                run_group_in_parallel(tasks)
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