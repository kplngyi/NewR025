import os
from pathlib import Path

import numpy as np
import torch


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


def create_balanced_sampler(targets):
    targets = np.asarray(targets)
    if targets.ndim != 1:
        raise ValueError(f"Expected 1D targets, got shape {targets.shape}")
    if targets.size == 0:
        raise ValueError("Cannot create a sampler for an empty target array")

    classes, counts = np.unique(targets, return_counts=True)
    class_weight_map = {
        cls: 1.0 / float(count) for cls, count in zip(classes.tolist(), counts.tolist())
    }
    sample_weights = np.asarray(
        [class_weight_map[target] for target in targets.tolist()], dtype=np.float32
    )
    sampler = TrackingWeightedRandomSampler(
        weights=sample_weights.astype(np.float64).tolist(),
        num_samples=int(targets.size),
        replacement=True,
        targets=targets,
    )
    class_counts = {int(cls): int(count) for cls, count in zip(classes, counts)}
    return sampler, class_counts


class TrackingWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(self, weights, num_samples, replacement, targets):
        super().__init__(
            weights=weights, num_samples=num_samples, replacement=replacement
        )
        self.targets = np.asarray(targets)
        self.last_indices = []
        self.last_class_counts = {}
        self.last_class_ratios = {}

    def __iter__(self):
        indices = list(super().__iter__())
        self.last_indices = indices
        sampled_targets = self.targets[np.asarray(indices, dtype=int)]
        classes, counts = np.unique(sampled_targets, return_counts=True)
        total = max(1, int(sampled_targets.size))
        self.last_class_counts = {
            int(cls): int(count)
            for cls, count in zip(classes.tolist(), counts.tolist())
        }
        self.last_class_ratios = {
            int(cls): float(count) / float(total)
            for cls, count in zip(classes.tolist(), counts.tolist())
        }
        return iter(indices)


class SampledClassRatioLogger:
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        sampler = getattr(net, "iterator_train__sampler", None)
        if sampler is None:
            return
        class_counts = getattr(sampler, "last_class_counts", None)
        class_ratios = getattr(sampler, "last_class_ratios", None)
        if not class_counts or not class_ratios:
            return
        ratio_text = ", ".join(
            f"class {cls}: count={class_counts[cls]}, ratio={class_ratios[cls]:.3f}"
            for cls in sorted(class_counts)
        )
        print(f"Sampled train distribution this epoch -> {ratio_text}")
