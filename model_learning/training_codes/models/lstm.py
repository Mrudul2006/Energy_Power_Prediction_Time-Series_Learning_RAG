"""
MODEL 7 - LSTM.

Sequence-to-vector LSTM for multi-step energy forecasting. The module includes
the raw nn.Module and a small univariate training wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .common import ForecastResult
from .neural_common import TorchForecasterMixin, TorchTrainConfig, require_torch

torch, nn, F, _, _ = require_torch()


@dataclass
class LSTMConfig:
    input_size: int = 168
    horizon: int = 24
    input_dim: int = 1
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    bidirectional: bool = False


class LSTMForecastNet(nn.Module):
    """LSTM encoder with a dense direct multi-horizon head."""

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        direction_mult = 2 if config.bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * direction_mult),
            nn.Linear(config.hidden_dim * direction_mult, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.horizon),
        )

    def forward(self, x):
        """x shape: [batch, input_size, input_dim]."""

        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class LSTMForecaster(TorchForecasterMixin):
    """Trainable univariate LSTM wrapper."""

    def __init__(self, model_config: LSTMConfig | None = None, train_config: TorchTrainConfig | None = None) -> None:
        self.model_config = model_config or LSTMConfig()
        self.train_config = train_config or TorchTrainConfig(
            input_size=self.model_config.input_size,
            horizon=self.model_config.horizon,
        )
        self.model = LSTMForecastNet(self.model_config)
        self.scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        result = super().forecast(horizon=horizon)
        result.metadata.update({"model": "LSTM", "config": self.model_config.__dict__})
        return result
