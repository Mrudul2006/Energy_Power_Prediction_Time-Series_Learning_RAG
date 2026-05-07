"""
MODEL 15 - Mamba-style selective state space model.

This is a forecasting-oriented selective SSM block inspired by Mamba. It uses
input-dependent B, C, and delta parameters and a linear-time recurrent scan.
It is not a drop-in replacement for the official optimized Mamba kernels, but
it captures the core selective state-space mechanism in plain PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .common import ForecastResult
from .neural_common import TorchForecasterMixin, TorchTrainConfig, require_torch

torch, nn, F, _, _ = require_torch()


@dataclass
class MambaConfig:
    input_size: int = 720
    horizon: int = 168
    input_dim: int = 1
    d_model: int = 96
    num_layers: int = 4
    expansion: int = 2
    dropout: float = 0.05


class SelectiveSSMBlock(nn.Module):
    """Plain PyTorch selective SSM scan."""

    def __init__(self, d_model: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.inner_dim = d_model * expansion
        self.in_proj = nn.Linear(d_model, self.inner_dim)
        self.delta_proj = nn.Linear(d_model, self.inner_dim)
        self.b_proj = nn.Linear(d_model, self.inner_dim)
        self.c_proj = nn.Linear(d_model, self.inner_dim)
        self.a_log = nn.Parameter(torch.zeros(self.inner_dim))
        self.d_skip = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, length, _ = x.shape
        u = torch.tanh(self.in_proj(x))
        delta = F.softplus(self.delta_proj(x)) + 1e-4
        b_t = torch.tanh(self.b_proj(x))
        c_t = torch.tanh(self.c_proj(x))
        a = -torch.exp(self.a_log).view(1, -1)
        state = x.new_zeros((bsz, self.inner_dim))
        outputs = []
        for t in range(length):
            decay = torch.exp(delta[:, t, :] * a)
            state = decay * state + delta[:, t, :] * b_t[:, t, :] * u[:, t, :]
            y_t = c_t[:, t, :] * state + self.d_skip.view(1, -1) * u[:, t, :]
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        return self.norm(x + self.dropout(self.out_proj(y)))


class MambaForecastNet(nn.Module):
    """Stacked selective SSM forecaster."""

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.input_dim, config.d_model)
        self.blocks = nn.Sequential(
            *[SelectiveSSMBlock(config.d_model, config.expansion, config.dropout) for _ in range(config.num_layers)]
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(config.input_size * config.d_model),
            nn.Linear(config.input_size * config.d_model, config.horizon),
        )

    def forward(self, x):
        z = self.embedding(x)
        z = self.blocks(z)
        return self.head(z)


class MambaForecaster(TorchForecasterMixin):
    """Trainable univariate Mamba-style wrapper."""

    def __init__(self, model_config: MambaConfig | None = None, train_config: TorchTrainConfig | None = None) -> None:
        self.model_config = model_config or MambaConfig()
        self.train_config = train_config or TorchTrainConfig(
            input_size=self.model_config.input_size,
            horizon=self.model_config.horizon,
        )
        self.model = MambaForecastNet(self.model_config)
        self.scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        result = super().forecast(horizon=horizon)
        result.metadata.update({"model": "MambaSelectiveSSM", "config": self.model_config.__dict__})
        return result
