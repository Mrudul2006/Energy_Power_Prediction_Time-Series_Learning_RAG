"""
MODEL 8 - Temporal Convolutional Network (TCN).

Implements dilated causal convolutions with residual connections. Dilation can
be exponential or explicitly set to energy-relevant lags such as 1, 2, 4, 8,
24, 48, 168.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .common import ForecastResult
from .neural_common import TorchForecasterMixin, TorchTrainConfig, require_torch

torch, nn, F, _, _ = require_torch()


@dataclass
class TCNConfig:
    input_size: int = 336
    horizon: int = 24
    input_dim: int = 1
    channels: int = 96
    kernel_size: int = 3
    dilations: Sequence[int] = (1, 2, 4, 8, 16, 24, 48, 96, 168)
    dropout: float = 0.10


class CausalConv1d(nn.Module):
    """Left-padded 1D convolution that never sees the future."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TCNForecastNet(nn.Module):
    """Dilated causal-convolution direct forecaster."""

    def __init__(self, config: TCNConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(config.input_dim, config.channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TCNResidualBlock(config.channels, config.kernel_size, int(dilation), config.dropout)
                for dilation in config.dilations
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(config.channels),
            nn.Linear(config.channels, config.channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.channels, config.horizon),
        )

    def forward(self, x):
        """x shape: [batch, input_size, input_dim]."""

        z = x.transpose(1, 2)
        z = self.input_projection(z)
        z = self.blocks(z)
        return self.head(z)


class TCNForecaster(TorchForecasterMixin):
    """Trainable univariate TCN wrapper."""

    def __init__(self, model_config: TCNConfig | None = None, train_config: TorchTrainConfig | None = None) -> None:
        self.model_config = model_config or TCNConfig()
        self.train_config = train_config or TorchTrainConfig(
            input_size=self.model_config.input_size,
            horizon=self.model_config.horizon,
        )
        self.model = TCNForecastNet(self.model_config)
        self.scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        result = super().forecast(horizon=horizon)
        result.metadata.update({"model": "TCN", "config": self.model_config.__dict__})
        return result
