import torch
import torch.nn as nn

from braindecode.models import ShallowFBCSPNet


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1)
        return x * s


class TemporalSEConvNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout=0.5, base_filters=32):
        super().__init__()
        hidden = base_filters * 2
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, base_filters, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ELU(),
            nn.Conv1d(
                base_filters,
                base_filters,
                kernel_size=9,
                padding=4,
                groups=base_filters,
                bias=False,
            ),
            nn.Conv1d(base_filters, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            SEBlock1D(hidden),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(-1)
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, channels, time), got {tuple(x.shape)}"
            )
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


class TemporalSEEncoder(nn.Module):
    def __init__(self, in_channels, dropout=0.5, base_filters=32):
        super().__init__()
        hidden = base_filters * 2
        self.hidden = hidden
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ELU(),
            nn.Conv1d(
                base_filters,
                base_filters,
                kernel_size=9,
                padding=4,
                groups=base_filters,
                bias=False,
            ),
            nn.Conv1d(base_filters, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            SEBlock1D(hidden),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.features(x).squeeze(-1)


class FusionTemporalSEConvNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        eeg_channels,
        fnirs_channels,
        dropout=0.5,
        base_filters=32,
        fusion_hidden=128,
    ):
        super().__init__()
        if eeg_channels <= 0 or fnirs_channels <= 0:
            raise ValueError("fusion_temporal_se 需要同时包含 EEG 和 fNIRS 通道。")
        if eeg_channels + fnirs_channels != n_channels:
            raise ValueError(
                f"Expected eeg_channels + fnirs_channels == n_channels, got {eeg_channels} + {fnirs_channels} != {n_channels}"
            )

        self.eeg_channels = eeg_channels
        self.fnirs_channels = fnirs_channels
        self.eeg_encoder = TemporalSEEncoder(
            in_channels=eeg_channels,
            dropout=dropout,
            base_filters=base_filters,
        )
        self.fnirs_encoder = TemporalSEEncoder(
            in_channels=fnirs_channels,
            dropout=dropout,
            base_filters=base_filters,
        )
        fusion_in = self.eeg_encoder.hidden + self.fnirs_encoder.hidden
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, n_classes),
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(-1)
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, channels, time), got {tuple(x.shape)}"
            )

        eeg_x = x[:, : self.eeg_channels, :]
        fnirs_x = x[:, self.eeg_channels :, :]
        eeg_emb = self.eeg_encoder(eeg_x)
        fnirs_emb = self.fnirs_encoder(fnirs_x)
        fused_emb = torch.cat([eeg_emb, fnirs_emb], dim=1)
        return self.classifier(fused_emb)


def build_model(
    model_name,
    n_channels,
    n_classes,
    n_times,
    eeg_channels=None,
    fnirs_channels=None,
):
    name = (model_name or "shallow").strip().lower()
    if name == "shallow":
        return ShallowFBCSPNet(
            n_channels,
            n_classes,
            n_times=n_times,
            final_conv_length="auto",
        )
    if name == "temporal_se":
        return TemporalSEConvNet(n_channels=n_channels, n_classes=n_classes)
    if name == "fusion_temporal_se":
        if eeg_channels is None or fnirs_channels is None:
            raise ValueError(
                "fusion_temporal_se 需要显式提供 eeg_channels 和 fnirs_channels。"
            )
        return FusionTemporalSEConvNet(
            n_channels=n_channels,
            n_classes=n_classes,
            eeg_channels=eeg_channels,
            fnirs_channels=fnirs_channels,
        )
    raise ValueError(
        f"Unsupported model '{model_name}'. Use one of: shallow, temporal_se, fusion_temporal_se"
    )
