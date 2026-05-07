"""
MODEL 10 - N-HiTS.

Hierarchical interpolation architecture. Each block sees the residual at a
different temporal resolution, predicts a backcast at input resolution, and
predicts coarse forecast knots that are interpolated to the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .common import ForecastResult
from .neural_common import TorchForecasterMixin, TorchTrainConfig, require_torch

torch, nn, F, _, _ = require_torch()


@dataclass
class NHITSConfig:
    input_size: int = 720
    horizon: int = 168
    pooling_rates: Sequence[int] = (1, 2, 4, 8)
    interpolation_sizes: Sequence[int] = (168, 84, 42, 21)
    blocks_per_stack: int = 1
    hidden_dim: int = 256
    layers: int = 3
    dropout: float = 0.05
    interpolation_mode: str = "linear"


class NHITSBlock(nn.Module):
    def __init__(self, input_size: int, horizon: int, pooling_rate: int, interpolation_size: int, config: NHITSConfig) -> None:
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.pooling_rate = int(pooling_rate)
        self.interpolation_size = int(interpolation_size)
        pooled_size = max(1, input_size // self.pooling_rate)
        layers = []
        in_dim = pooled_size
        for _ in range(config.layers):
            layers.extend([nn.Linear(in_dim, config.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = config.hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(config.hidden_dim, input_size)
        self.forecast_knots = nn.Linear(config.hidden_dim, self.interpolation_size)
        self.mode = config.interpolation_mode

    def _pool(self, x):
        if self.pooling_rate <= 1:
            return x
        pooled = F.avg_pool1d(x.unsqueeze(1), kernel_size=self.pooling_rate, stride=self.pooling_rate, ceil_mode=False)
        return pooled.squeeze(1)

    def _interpolate(self, knots):
        z = knots.unsqueeze(1)
        if self.mode == "nearest":
            out = F.interpolate(z, size=self.horizon, mode="nearest")
        else:
            out = F.interpolate(z, size=self.horizon, mode=self.mode, align_corners=False)
        return out.squeeze(1)

    def forward(self, x):
        pooled = self._pool(x)
        z = self.mlp(pooled)
        backcast = self.backcast_head(z)
        forecast = self._interpolate(self.forecast_knots(z))
        return backcast, forecast


class NHITSForecastNet(nn.Module):
    """Stacked N-HiTS network."""

    def __init__(self, config: NHITSConfig) -> None:
        super().__init__()
        self.config = config
        blocks = []
        rates = list(config.pooling_rates)
        sizes = list(config.interpolation_sizes)
        if len(sizes) != len(rates):
            raise ValueError("interpolation_sizes must match pooling_rates length")
        for rate, size in zip(rates, sizes):
            for _ in range(config.blocks_per_stack):
                blocks.append(NHITSBlock(config.input_size, config.horizon, rate, size, config))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        residual = x.squeeze(-1)
        forecast = residual.new_zeros((residual.shape[0], self.config.horizon))
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


class NHITSForecaster(TorchForecasterMixin):
    """Trainable univariate N-HiTS wrapper."""

    def __init__(self, model_config: NHITSConfig | None = None, train_config: TorchTrainConfig | None = None) -> None:
        self.model_config = model_config or NHITSConfig()
        self.train_config = train_config or TorchTrainConfig(
            input_size=self.model_config.input_size,
            horizon=self.model_config.horizon,
        )
        self.model = NHITSForecastNet(self.model_config)
        self.scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        result = super().forecast(horizon=horizon)
        result.metadata.update({"model": "N-HiTS", "config": self.model_config.__dict__})
        return result
