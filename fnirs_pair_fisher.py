import numpy as np


def normalize_score_vector(values):
    values = np.asarray(values, dtype=float)
    v_min = values.min()
    v_max = values.max()
    if v_max - v_min < 1e-12:
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def zscore_vector(values):
    values = np.asarray(values, dtype=float)
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def compute_slope_feature(signal_matrix):
    time_axis = np.arange(signal_matrix.shape[1], dtype=float)
    time_centered = time_axis - time_axis.mean()
    denom = np.dot(time_centered, time_centered)
    if denom == 0:
        return np.zeros(signal_matrix.shape[0], dtype=float)
    return signal_matrix @ time_centered / denom


def split_fnirs_channel_pairs(ch_names):
    lookup = {}
    order = []
    for idx, name in enumerate(ch_names):
        key = name.lower().strip()
        if key.endswith(" hbo"):
            pair_key = key[:-4]
            entry = lookup.setdefault(
                pair_key, {"hbo_idx": None, "hbr_idx": None, "pair_name": pair_key}
            )
            entry["hbo_idx"] = idx
            order.append(pair_key) if pair_key not in order else None
        elif key.endswith(" hbr"):
            pair_key = key[:-4]
            entry = lookup.setdefault(
                pair_key, {"hbo_idx": None, "hbr_idx": None, "pair_name": pair_key}
            )
            entry["hbr_idx"] = idx
            order.append(pair_key) if pair_key not in order else None

    pair_infos = []
    for pair_key in order:
        entry = lookup[pair_key]
        if entry["hbo_idx"] is None or entry["hbr_idx"] is None:
            continue
        pair_infos.append(
            {
                "pair_key": pair_key,
                "pair_name": pair_key,
                "hbo_idx": entry["hbo_idx"],
                "hbr_idx": entry["hbr_idx"],
            }
        )
    if not pair_infos:
        raise ValueError("No HbO/HbR pairs found in channel names.")
    return pair_infos


def compute_pair_fisher_scores(
    windows_dataset,
    ch_names,
    hbo_weight=0.7,
    feature_weights=(0.5, 0.2, 0.3),
):
    n_windows = len(windows_dataset)
    if n_windows == 0:
        raise ValueError("windows_dataset is empty")

    sample_X, _, _ = windows_dataset[0]
    sample_X = np.asarray(sample_X)
    pair_infos = split_fnirs_channel_pairs(ch_names)
    n_pairs = len(pair_infos)

    y_all = np.zeros(n_windows, dtype=int)
    hbo_mean = np.zeros((n_windows, n_pairs), dtype=float)
    hbo_std = np.zeros((n_windows, n_pairs), dtype=float)
    hbo_slope = np.zeros((n_windows, n_pairs), dtype=float)
    hbr_mean = np.zeros((n_windows, n_pairs), dtype=float)
    hbr_std = np.zeros((n_windows, n_pairs), dtype=float)
    hbr_slope = np.zeros((n_windows, n_pairs), dtype=float)

    for i, (X_i, y_i, _) in enumerate(windows_dataset):
        X_i = np.asarray(X_i)
        if X_i.shape[0] != sample_X.shape[0]:
            raise ValueError(
                f"Window {i}: expected {sample_X.shape[0]} channels but got {X_i.shape[0]}"
            )
        y_all[i] = int(y_i)
        for pair_pos, pair_info in enumerate(pair_infos):
            hbo_signal = X_i[pair_info["hbo_idx"]]
            hbr_signal = X_i[pair_info["hbr_idx"]]
            hbo_mean[i, pair_pos] = np.mean(hbo_signal)
            hbo_std[i, pair_pos] = np.std(hbo_signal)
            hbr_mean[i, pair_pos] = np.mean(hbr_signal)
            hbr_std[i, pair_pos] = np.std(hbr_signal)

        hbo_signals = X_i[[info["hbo_idx"] for info in pair_infos], :]
        hbr_signals = X_i[[info["hbr_idx"] for info in pair_infos], :]
        hbo_slope[i, :] = compute_slope_feature(hbo_signals)
        hbr_slope[i, :] = compute_slope_feature(hbr_signals)

    def fisher_scores(feature_matrix):
        classes = np.unique(y_all)
        mu_total = feature_matrix.mean(axis=0)
        Sb = np.zeros(feature_matrix.shape[1], dtype=float)
        Sw = np.zeros(feature_matrix.shape[1], dtype=float)
        for c in classes:
            Fc = feature_matrix[y_all == c]
            nc = Fc.shape[0]
            if nc <= 1:
                continue
            muc = Fc.mean(axis=0)
            varc = Fc.var(axis=0)
            Sb += nc * (muc - mu_total) ** 2
            Sw += nc * varc
        return Sb / (Sw + 1e-8)

    hbo_scores = (
        feature_weights[0] * zscore_vector(fisher_scores(hbo_mean))
        + feature_weights[1] * zscore_vector(fisher_scores(hbo_std))
        + feature_weights[2] * zscore_vector(fisher_scores(hbo_slope))
    )
    hbr_scores = (
        feature_weights[0] * zscore_vector(fisher_scores(hbr_mean))
        + feature_weights[1] * zscore_vector(fisher_scores(hbr_std))
        + feature_weights[2] * zscore_vector(fisher_scores(hbr_slope))
    )

    pair_scores = normalize_score_vector(
        hbo_weight * hbo_scores + (1 - hbo_weight) * hbr_scores
    )
    rank_idx = np.argsort(pair_scores)[::-1]

    detail_rows = []
    for pair_pos, pair_info in enumerate(pair_infos):
        detail_rows.append(
            {
                "pair_idx": pair_pos,
                "pair_name": pair_info["pair_name"],
                "hbo_idx": pair_info["hbo_idx"],
                "hbr_idx": pair_info["hbr_idx"],
                "hbo_score": float(hbo_scores[pair_pos]),
                "hbr_score": float(hbr_scores[pair_pos]),
                "pair_score": float(pair_scores[pair_pos]),
            }
        )
    return rank_idx, pair_scores, pair_infos, detail_rows


def expand_pair_selection_to_channels(
    selected_pair_indices, pair_infos, channel_offset=0
):
    selected_channels = []
    rows = []
    for pair_idx in selected_pair_indices:
        pair_info = pair_infos[pair_idx]
        selected_channels.extend(
            [
                channel_offset + pair_info["hbo_idx"],
                channel_offset + pair_info["hbr_idx"],
            ]
        )
        rows.append(pair_info)
    return selected_channels, rows
