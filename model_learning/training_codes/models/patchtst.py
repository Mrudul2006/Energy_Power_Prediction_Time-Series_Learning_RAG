"""
MODEL 12 - PatchTST.

PatchTST tokenizes a long sequence into patches so attention is paid between
patches instead of individual time steps. For hourly energy data, patch_len=24
turns each token into a daily load profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .common import ForecastResult
from .neural_common import TorchForecasterMixin, TorchTrainConfig, require_torch

torch, nn, F, _, _ = require_torch()


@dataclass
class PatchTSTConfig:
    input_size: int = 720
    horizon: int = 168
    input_dim: int = 1
    patch_len: int = 24
    stride: int = 24
    d_model: int = 128
    n_heads: int = 8
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.10
    target_channel: int = 0


class PatchTSTForecastNet(nn.Module):
    """Channel-independent patched Transformer forecaster."""

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.config = config
        if config.input_size < config.patch_len:
            raise ValueError("input_size must be at least patch_len")
        self.n_patches = 1 + (config.input_size - config.patch_len) // config.stride
        self.patch_projection = nn.Linear(config.patch_len, config.d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.n_patches * config.d_model),
            nn.Linear(self.n_patches * config.d_model, config.horizon),
        )

    def forward(self, x):
        """x shape: [batch, input_size, channels]."""

        bsz, _, channels = x.shape
        z = x.transpose(1, 2)
        patches = z.unfold(dimension=-1, size=self.config.patch_len, step=self.config.stride)
        patches = patches[:, :, : self.n_patches, :]
        tokens = patches.reshape(bsz * channels, self.n_patches, self.config.patch_len)
        tokens = self.patch_projection(tokens) + self.position
        encoded = self.encoder(tokens)
        out = self.head(encoded).reshape(bsz, channels, self.config.horizon)
        channel = min(self.config.target_channel, channels - 1)
        return out[:, channel, :]


class PatchTSTForecaster(TorchForecasterMixin):
    """Trainable univariate PatchTST wrapper."""

    def __init__(self, model_config: PatchTSTConfig | None = None, train_config: TorchTrainConfig | None = None) -> None:
        self.model_config = model_config or PatchTSTConfig()
        self.train_config = train_config or TorchTrainConfig(
            input_size=self.model_config.input_size,
            horizon=self.model_config.horizon,
        )
        self.model = PatchTSTForecastNet(self.model_config)
        self.scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        result = super().forecast(horizon=horizon)
        result.metadata.update({"model": "PatchTST", "config": self.model_config.__dict__})
        return result
