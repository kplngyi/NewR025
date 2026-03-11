import torch
import torch.nn as nn

from braindecode.models import Deep4Net, EEGNet, ShallowFBCSPNet


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


def build_model(model_name, n_channels, n_classes, n_times):
    name = (model_name or "shallow").strip().lower()
    if name == "shallow":
        return ShallowFBCSPNet(
            n_channels,
            n_classes,
            n_times=n_times,
            final_conv_length="auto",
        )
    if name == "eegnet":
        return EEGNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto",
        )
    if name == "deep4":
        return Deep4Net(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto",
        )
    if name == "temporal_se":
        return TemporalSEConvNet(n_channels=n_channels, n_classes=n_classes)
    raise ValueError(
        f"Unsupported model '{model_name}'. Use one of: shallow, temporal_se, eegnet, deep4"
    )
