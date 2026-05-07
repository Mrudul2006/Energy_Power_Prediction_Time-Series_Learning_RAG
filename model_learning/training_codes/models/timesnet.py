"""
MODEL 14 - TimesNet.

TimesNet reshapes 1D sequences into 2D period-by-cycle grids and applies 2D
convolutions to capture intra-period and inter-period structure together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .common import ForecastResult
from .neural_common import TorchForecasterMixin, TorchTrainConfig, require_torch

torch, nn, F, _, _ = require_torch()


@dataclass
class TimesNetConfig:
    input_size: int = 720
    horizon: int = 168
    input_dim: int = 1
    d_model: int = 64
    periods: Sequence[int] = (24, 168)
    num_blocks: int = 3
    kernel_sizes: Sequence[int] = (1, 3, 5)
    dropout: float = 0.10


class Inception2D(nn.Module):
    def __init__(self, channels: int, kernel_sizes: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                    nn.Dropout2d(dropout),
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x):
        return torch.stack([conv(x) for conv in self.convs], dim=0).mean(dim=0)


class TimesBlock(nn.Module):
    def __init__(self, config: TimesNetConfig) -> None:
        super().__init__()
        self.config = config
        self.conv = Inception2D(config.d_model, config.kernel_sizes, config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

    def _period_conv(self, x, period: int):
        bsz, length, channels = x.shape
        pad_len = (period - length % period) % period
        if pad_len:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        total = x_pad.shape[1]
        grid = x_pad.reshape(bsz, total // period, period, channels).permute(0, 3, 1, 2)
        out = self.conv(grid)
        out = out.permute(0, 2, 3, 1).reshape(bsz, total, channels)
        return out[:, :length, :]

    def forward(self, x):
        period_outputs = [self._period_conv(x, int(period)) for period in self.config.periods]
        z = torch.stack(period_outputs, dim=0).mean(dim=0)
        return self.norm(x + z)


class TimesNetForecastNet(nn.Module):
    """2D-periodic convolutional forecaster."""

    def __init__(self, config: TimesNetConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.input_dim, config.d_model)
        self.blocks = nn.Sequential(*[TimesBlock(config) for _ in range(config.num_blocks)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(config.input_size * config.d_model),
            nn.Linear(config.input_size * config.d_model, config.horizon),
        )

    def forward(self, x):
        z = self.embedding(x)
        z = self.blocks(z)
        return self.head(z)


class TimesNetForecaster(TorchForecasterMixin):
    """Trainable univariate TimesNet wrapper."""

    def __init__(self, model_config: TimesNetConfig | None = None, train_config: TorchTrainConfig | None = None) -> None:
        self.model_config = model_config or TimesNetConfig()
        self.train_config = train_config or TorchTrainConfig(
            input_size=self.model_config.input_size,
            horizon=self.model_config.horizon,
        )
        self.model = TimesNetForecastNet(self.model_config)
        self.scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        result = super().forecast(horizon=horizon)
        result.metadata.update({"model": "TimesNet", "config": self.model_config.__dict__})
        return result
