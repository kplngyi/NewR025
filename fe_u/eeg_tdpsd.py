import numpy as np


def normalize_score_vector(scores):
    scores = np.asarray(scores, dtype=float)
    score_min = scores.min()
    score_max = scores.max()
    if score_max - score_min < 1e-12:
        return np.zeros_like(scores)
    return (scores - score_min) / (score_max - score_min)


def tdpsd_features_1d(x, eps=1e-10):
    """
    单通道 TDPSD 特征:
        f1 = log(var(x))
        f2 = log(mean(diff(x)^2))
        f3 = log(mean(diff(diff(x))^2))
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    if x.size < 3:
        raise ValueError(
            "Signal length must be at least 3 for TDPSD feature extraction."
        )

    var_x = np.var(x)

    d1 = np.diff(x)
    d2 = np.diff(d1)

    e1 = np.mean(d1**2)
    e2 = np.mean(d2**2)

    return np.array(
        [
            np.log(var_x + eps),
            np.log(e1 + eps),
            np.log(e2 + eps),
        ],
        dtype=float,
    )


def fisher_score_2d(F, y):
    """
    对二维特征矩阵计算每列的 Fisher score
    F: shape (n_samples, n_features)
    y: shape (n_samples,)
    """
    F = np.asarray(F, dtype=float)
    y = np.asarray(y)

    classes = np.unique(y)
    mu_total = F.mean(axis=0)

    Sb = np.zeros(F.shape[1], dtype=float)
    Sw = np.zeros(F.shape[1], dtype=float)

    for c in classes:
        Fc = F[y == c]
        nc = Fc.shape[0]
        if nc <= 1:
            continue

        muc = Fc.mean(axis=0)
        varc = Fc.var(axis=0, ddof=1)

        Sb += nc * (muc - mu_total) ** 2
        Sw += nc * varc

    return Sb / (Sw + 1e-8)


def fisher_score_channels_from_windows_dataset_tdpsd(windows_dataset):
    """
    使用 TDPSD 特征进行通道选择。
    对每个通道提取 3 个 TDPSD 特征，分别计算 Fisher score，
    最后对该通道的 3 个分数取sum作为通道总分。

    windows_dataset[i] -> (X, y, meta), X shape = (n_ch, n_time)

    返回:
        rank_idx: 通道索引排序（从大到小）
        scores:   每个通道最终分数
        sub_scores: shape (n_channels, 3)，每个通道 3 个 TDPSD 子特征的分数
    """
    n_windows = len(windows_dataset)
    if n_windows == 0:
        raise ValueError("windows_dataset is empty")

    sample_X, _, _ = windows_dataset[0]
    sample_X = np.asarray(sample_X, dtype=float)
    if sample_X.ndim != 2:
        raise ValueError(f"Expected X shape (n_ch, n_time), got {sample_X.shape}")

    n_channels = sample_X.shape[0]
    n_tdpsd = 3

    # shape: (n_windows, n_channels, 3)
    F_all = np.zeros((n_windows, n_channels, n_tdpsd), dtype=float)
    ys = np.zeros((n_windows,), dtype=int)

    for i in range(n_windows):
        X_i, y_i, _ = windows_dataset[i]
        X_i = np.asarray(X_i, dtype=float)

        if X_i.ndim != 2:
            raise ValueError(
                f"Window {i}: expected X_i shape (n_ch, n_time), got {X_i.shape}"
            )
        if X_i.shape[0] != n_channels:
            raise ValueError(
                f"Inconsistent channel count at window {i}: {X_i.shape[0]} vs {n_channels}"
            )

        for ch in range(n_channels):
            F_all[i, ch, :] = tdpsd_features_1d(X_i[ch])

        ys[i] = int(y_i)

    # 对每个 TDPSD 子特征分别做通道级 Fisher score
    sub_scores = np.zeros((n_channels, n_tdpsd), dtype=float)
    for k in range(n_tdpsd):
        # F_k shape: (n_windows, n_channels)
        F_k = F_all[:, :, k]
        sub_scores[:, k] = fisher_score_2d(F_k, ys)

    # 先把每个子特征的 Fisher 分数归一化，再融合成通道总分，避免某一类
    # 子特征数值尺度过大而主导通道排序。
    sub_scores_norm = np.zeros_like(sub_scores)
    for k in range(n_tdpsd):
        sub_scores_norm[:, k] = normalize_score_vector(sub_scores[:, k])

    scores = normalize_score_vector(sub_scores_norm.sum(axis=1))
    rank_idx = np.argsort(scores)[::-1]

    return rank_idx, scores, sub_scores_norm
