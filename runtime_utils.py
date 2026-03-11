import os
from pathlib import Path

import numpy as np


def parse_known_args(parser):
    # Ignore notebook/kernel flags injected by Jupyter or Colab.
    args, _ = parser.parse_known_args()
    return args


def resolve_project_root(script_file, project_root=None):
    if project_root:
        return Path(project_root).expanduser().resolve()
    env_root = os.environ.get("ROBIO_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(script_file).resolve().parent


def resolve_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir) / path).resolve()


def prepare_runtime_dirs(project_root, output_root=None):
    output_dir = (
        resolve_path(output_root, project_root) if output_root else Path(project_root)
    )
    logs_dir = output_dir / "Logs"
    cache_dir = output_dir / ".cache"
    mplconfig_dir = output_dir / ".mplconfig"
    logs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mplconfig_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

    return {
        "output_dir": output_dir,
        "logs_dir": logs_dir,
        "cache_dir": cache_dir,
        "mplconfig_dir": mplconfig_dir,
    }


def add_common_runtime_args(parser):
    parser.add_argument("--project_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--top_k_step", type=int, default=2)
    parser.add_argument("--min_top_k", type=int, default=None)
    parser.add_argument("--files_limit", type=int, default=None)
    return parser


def resolve_requested_top_k(arg_top_k, config_top_k, max_channels):
    requested_top_k = arg_top_k if arg_top_k is not None else config_top_k
    if requested_top_k is None:
        return int(max_channels)
    requested_top_k = int(requested_top_k)
    if requested_top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {requested_top_k}")
    return min(requested_top_k, int(max_channels))


def normalize_channel_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores, got shape {scores.shape}")
    if scores.size == 0:
        return scores.copy()
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    if score_max - score_min < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - score_min) / (score_max - score_min)
