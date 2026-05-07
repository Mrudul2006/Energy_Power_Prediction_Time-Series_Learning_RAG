"""
MODEL 17 - DLinear and NLinear.

These deceptively simple baselines are important controls for long-horizon
forecasting. DLinear decomposes trend and seasonal/remainder parts with a
moving average; NLinear subtracts the last value before a linear projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .common import ForecastResult
from .neural_common import TorchForecasterMixin, TorchTrainConfig, require_torch

torch, nn, F, _, _ = require_torch()


@dataclass
class LinearBaselineConfig:
    input_size: int = 336
    horizon: int = 168
    input_dim: int = 1
    model_type: Literal["dlinear", "nlinear"] = "dlinear"
    moving_average: int = 25
    individual: bool = False


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, 0:1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.avg(padded.transpose(1, 2)).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearNet(nn.Module):
    def __init__(self, config: LinearBaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.decomposition = SeriesDecomposition(config.moving_average)
        self.seasonal = nn.Linear(config.input_size, config.horizon)
        self.trend = nn.Linear(config.input_size, config.horizon)

    def forward(self, x):
        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)
        out = self.seasonal(seasonal) + self.trend(trend)
        return out[:, 0, :]


class NLinearNet(nn.Module):
    def __init__(self, config: LinearBaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.input_size, config.horizon)

    def forward(self, x):
        last = x[:, -1:, :]
        centered = x - last
        z = centered.transpose(1, 2)
        out = self.linear(z).transpose(1, 2)
        return (out + last).squeeze(-1)


class LinearBaselineForecaster(TorchForecasterMixin):
    """Trainable DLinear or NLinear univariate wrapper."""

    def __init__(
        self,
        model_config: LinearBaselineConfig | None = None,
        train_config: TorchTrainConfig | None = None,
    ) -> None:
        self.model_config = model_config or LinearBaselineConfig()
        self.train_config = train_config or TorchTrainConfig(
            input_size=self.model_config.input_size,
            horizon=self.model_config.horizon,
            epochs=20,
        )
        self.model = DLinearNet(self.model_config) if self.model_config.model_type == "dlinear" else NLinearNet(self.model_config)
        self.scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        result = super().forecast(horizon=horizon)
        result.metadata.update({"model": self.model_config.model_type.upper(), "config": self.model_config.__dict__})
        return result
