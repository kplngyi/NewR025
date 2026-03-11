import numpy as np
from scipy.signal import welch


def bandpower_welch(x, fs, fmin, fmax, nperseg=None, log_power=True, eps=1e-10):
    """
    Estimate 1D signal bandpower with Welch PSD and integrate within [fmin, fmax).
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    if x.size == 0:
        raise ValueError("Input signal is empty.")

    if nperseg is None:
        nperseg = min(x.size, int(fs * 2))

    freqs, psd = welch(x, fs=fs, nperseg=nperseg)

    idx = (freqs >= fmin) & (freqs < fmax)
    if not np.any(idx):
        bp = 0.0
    else:
        bp = np.trapezoid(psd[idx], freqs[idx])

    if log_power:
        bp = np.log(bp + eps)

    return float(bp)


def fisher_score_channels_alpha_beta_from_windows_dataset(windows_dataset, fs, mode="avg"):
    """
    Rank channels by Fisher score using EEG alpha/beta bandpower features.

    windows_dataset[i] -> (X, y, meta), X shape = (n_ch, n_time)

    mode:
        "alpha": alpha bandpower only
        "beta": beta bandpower only
        "avg": average of alpha and beta bandpower
    """
    n_windows = len(windows_dataset)
    if n_windows == 0:
        raise ValueError("windows_dataset is empty")

    sample_X, _, _ = windows_dataset[0]
    sample_X = np.asarray(sample_X, dtype=float)
    if sample_X.ndim != 2:
        raise ValueError(f"Expected X shape (n_ch, n_time), got {sample_X.shape}")

    n_channels = sample_X.shape[0]
    F = np.zeros((n_windows, n_channels), dtype=float)
    ys = np.zeros((n_windows,), dtype=int)

    for i in range(n_windows):
        X_i, y_i, _ = windows_dataset[i]
        X_i = np.asarray(X_i, dtype=float)

        if X_i.ndim != 2:
            raise ValueError(f"Window {i}: expected X_i shape (n_ch, n_time), got {X_i.shape}")
        if X_i.shape[0] != n_channels:
            raise ValueError(
                f"Inconsistent channel count at window {i}: {X_i.shape[0]} vs {n_channels}"
            )

        alpha_feat = np.zeros(n_channels, dtype=float)
        beta_feat = np.zeros(n_channels, dtype=float)

        for ch in range(n_channels):
            alpha_feat[ch] = bandpower_welch(X_i[ch], fs, 8, 13)
            beta_feat[ch] = bandpower_welch(X_i[ch], fs, 13, 30)

        if mode == "alpha":
            F[i, :] = alpha_feat
        elif mode == "beta":
            F[i, :] = beta_feat
        elif mode == "avg":
            F[i, :] = 0.5 * (alpha_feat + beta_feat)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        ys[i] = int(y_i)

    classes = np.unique(ys)
    mu_total = F.mean(axis=0)

    Sb = np.zeros(n_channels, dtype=float)
    Sw = np.zeros(n_channels, dtype=float)

    for c in classes:
        Xc = F[ys == c]
        nc = Xc.shape[0]
        if nc <= 1:
            continue

        muc = Xc.mean(axis=0)
        varc = Xc.var(axis=0, ddof=1)

        Sb += nc * (muc - mu_total) ** 2
        Sw += nc * varc

    scores = Sb / (Sw + 1e-8)
    rank_idx = np.argsort(scores)[::-1]
    return rank_idx, scores
