"""
MODEL 9 - N-BEATS.

Deep residual basis-expansion architecture for univariate forecasting. The
generic variant learns arbitrary backcast/forecast bases; the interpretable
variant uses polynomial trend and Fourier seasonal bases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence
import math

from .common import ForecastResult
from .neural_common import TorchForecasterMixin, TorchTrainConfig, require_torch

torch, nn, F, _, _ = require_torch()


@dataclass
class NBEATSConfig:
    input_size: int = 336
    horizon: int = 24
    stack_types: Sequence[str] = ("trend", "seasonality", "generic", "generic")
    blocks_per_stack: int = 2
    hidden_dim: int = 256
    layers: int = 4
    degree: int = 3
    harmonics: int = 12
    dropout: float = 0.0


class TrendBasis(nn.Module):
    def __init__(self, input_size: int, horizon: int, degree: int) -> None:
        super().__init__()
        backcast_t = torch.linspace(-1.0, 0.0, input_size)
        forecast_t = torch.linspace(0.0, 1.0, horizon)
        self.register_buffer("backcast_basis", torch.stack([backcast_t**i for i in range(degree + 1)]))
        self.register_buffer("forecast_basis", torch.stack([forecast_t**i for i in range(degree + 1)]))

    def forward(self, theta_b, theta_f):
        return theta_b @ self.backcast_basis, theta_f @ self.forecast_basis


class SeasonalityBasis(nn.Module):
    def __init__(self, input_size: int, horizon: int, harmonics: int) -> None:
        super().__init__()
        def build(length: int):
            t = torch.arange(length, dtype=torch.float32) / max(1, length)
            rows = []
            for k in range(1, harmonics + 1):
                rows.append(torch.sin(2.0 * math.pi * k * t))
                rows.append(torch.cos(2.0 * math.pi * k * t))
            return torch.stack(rows)

        self.register_buffer("backcast_basis", build(input_size))
        self.register_buffer("forecast_basis", build(horizon))

    def forward(self, theta_b, theta_f):
        return theta_b @ self.backcast_basis, theta_f @ self.forecast_basis


class NBEATSBlock(nn.Module):
    def __init__(self, input_size: int, horizon: int, block_type: str, config: NBEATSConfig) -> None:
        super().__init__()
        self.block_type = block_type
        layers = []
        in_dim = input_size
        for _ in range(config.layers):
            layers.extend([nn.Linear(in_dim, config.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = config.hidden_dim
        self.mlp = nn.Sequential(*layers)
        if block_type == "trend":
            theta_dim = config.degree + 1
            self.basis = TrendBasis(input_size, horizon, config.degree)
            self.theta = nn.Linear(config.hidden_dim, theta_dim * 2)
            self.backcast_linear = None
            self.forecast_linear = None
        elif block_type == "seasonality":
            theta_dim = config.harmonics * 2
            self.basis = SeasonalityBasis(input_size, horizon, config.harmonics)
            self.theta = nn.Linear(config.hidden_dim, theta_dim * 2)
            self.backcast_linear = None
            self.forecast_linear = None
        elif block_type == "generic":
            self.basis = None
            self.theta = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.backcast_linear = nn.Linear(config.hidden_dim, input_size)
            self.forecast_linear = nn.Linear(config.hidden_dim, horizon)
        else:
            raise ValueError("block_type must be trend, seasonality, or generic")

    def forward(self, x):
        z = self.mlp(x)
        theta = self.theta(z)
        if self.block_type == "generic":
            return self.backcast_linear(theta), self.forecast_linear(theta)
        theta_b, theta_f = torch.chunk(theta, 2, dim=-1)
        return self.basis(theta_b, theta_f)


class NBEATSForecastNet(nn.Module):
    """Stacked residual N-BEATS network."""

    def __init__(self, config: NBEATSConfig) -> None:
        super().__init__()
        self.config = config
        blocks = []
        for stack_type in config.stack_types:
            for _ in range(config.blocks_per_stack):
                blocks.append(NBEATSBlock(config.input_size, config.horizon, stack_type, config))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        residual = x.squeeze(-1)
        forecast = residual.new_zeros((residual.shape[0], self.config.horizon))
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


class NBEATSForecaster(TorchForecasterMixin):
    """Trainable univariate N-BEATS wrapper."""

    def __init__(self, model_config: NBEATSConfig | None = None, train_config: TorchTrainConfig | None = None) -> None:
        self.model_config = model_config or NBEATSConfig()
        self.train_config = train_config or TorchTrainConfig(
            input_size=self.model_config.input_size,
            horizon=self.model_config.horizon,
        )
        self.model = NBEATSForecastNet(self.model_config)
        self.scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        result = super().forecast(horizon=horizon)
        result.metadata.update({"model": "N-BEATS", "config": self.model_config.__dict__})
        return result
