"""
MODEL 16 - TimesFM-style foundation model wrapper.

The official Google TimesFM model requires external weights and package setup.
This module supports two practical paths:
1. Wrap a provided/preloaded TimesFM object for zero-shot forecasting.
2. Train a local patched Transformer fallback with the same "patches as tokens"
   inductive bias when the foundation model is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence
import os

import numpy as np

from .common import ForecastResult, StandardScaler1D, ensure_dir, ensure_series, future_index, save_json
from .neural_common import TorchTrainConfig, forecast_torch_univariate, require_torch, train_torch_forecaster

torch, nn, F, _, _ = require_torch()


@dataclass
class TimesFMConfig:
    context_len: int = 512
    horizon_len: int = 128
    patch_len: int = 32
    stride: int = 32
    d_model: int = 128
    n_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.10
    train_fallback: bool = True


class LocalTimesFMNet(nn.Module):
    """Small patched decoder-style Transformer used as a local fallback."""

    def __init__(self, config: TimesFMConfig) -> None:
        super().__init__()
        self.config = config
        self.n_patches = 1 + (config.context_len - config.patch_len) // config.stride
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
            nn.Linear(self.n_patches * config.d_model, config.horizon_len),
        )

    def forward(self, x):
        """x shape: [batch, context_len, 1]."""

        z = x.squeeze(-1)
        patches = z.unfold(dimension=-1, size=self.config.patch_len, step=self.config.stride)
        patches = patches[:, : self.n_patches, :]
        tokens = self.patch_projection(patches) + self.position
        encoded = self.encoder(tokens)
        return self.head(encoded)


class TimesFMForecaster:
    """
    Wrapper around a preloaded TimesFM-like object, with local fallback training.

    A supplied pretrained_model may expose either:
    - forecast(list_of_arrays, freq=...)
    - forecast(np.ndarray)
    - predict(np.ndarray)
    """

    def __init__(
        self,
        config: TimesFMConfig | None = None,
        pretrained_model: Optional[Any] = None,
        train_config: Optional[TorchTrainConfig] = None,
    ) -> None:
        self.config = config or TimesFMConfig()
        self.pretrained_model = pretrained_model
        self.train_config = train_config or TorchTrainConfig(
            input_size=self.config.context_len,
            horizon=self.config.horizon_len,
            epochs=10,
        )
        self.model = LocalTimesFMNet(self.config)
        self.scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None
        self.backend_ = "pretrained_timesfm" if pretrained_model is not None else "local_patch_transformer_fallback"

    def fit(self, data, target_col=None, timestamp_col=None) -> "TimesFMForecaster":
        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        self.series_ = y
        if self.pretrained_model is not None:
            return self
        if not self.config.train_fallback:
            return self
        fitted, scaler, losses = train_torch_forecaster(self.model, y.to_numpy(dtype=float), self.train_config)
        self.model = fitted
        self.scaler_ = scaler
        self.losses_ = losses
        return self

    def _forecast_pretrained(self, context: np.ndarray, horizon: int) -> np.ndarray:
        model = self.pretrained_model
        if hasattr(model, "forecast"):
            try:
                output = model.forecast([context], freq=[0])
            except TypeError:
                output = model.forecast(context.reshape(1, -1))
        elif hasattr(model, "predict"):
            output = model.predict(context.reshape(1, -1))
        else:
            raise TypeError("pretrained_model must expose forecast(...) or predict(...)")
        if isinstance(output, tuple):
            output = output[0]
        arr = np.asarray(output, dtype=float).reshape(-1)
        return arr[:horizon]

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        if self.series_ is None:
            raise RuntimeError("fit the forecaster before forecast")
        h = int(horizon or self.config.horizon_len)
        context = self.series_.to_numpy(dtype=float)[-self.config.context_len :]
        if len(context) < self.config.context_len:
            raise ValueError("series is shorter than context_len")
        if self.pretrained_model is not None:
            point = self._forecast_pretrained(context, h)
        else:
            if self.scaler_ is None:
                raise RuntimeError("fallback model was not trained; set train_fallback=True or provide pretrained_model")
            point = forecast_torch_univariate(self.model, self.series_.to_numpy(dtype=float), self.scaler_, self.train_config)
            point = point[:h]
        return ForecastResult(
            mean=point,
            index=future_index(self.series_.index, len(point)),
            metadata={
                "model": "TimesFM",
                "backend": self.backend_,
                "note": "local backend is not pretrained; provide official TimesFM weights for true zero-shot foundation behavior",
                "training_loss": self.losses_[-1] if self.losses_ else None,
            },
        )

    def save(self, path: str) -> None:
        ensure_dir(path)
        checkpoint_dir = os.path.join(path, "checkpoints")
        ensure_dir(checkpoint_dir)
        payload = {
            "state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "train_config": self.train_config.__dict__,
            "backend": self.backend_,
            "scaler": {"mean": float(self.scaler_.mean_), "scale": float(self.scaler_.scale_)} if self.scaler_ else None,
            "losses": list(self.losses_ or []),
        }
        torch.save(payload, os.path.join(checkpoint_dir, "final.pt"))
        save_json(
            {
                "model": "TimesFM",
                "config": payload["config"],
                "train_config": payload["train_config"],
                "backend": payload["backend"],
                "losses": payload["losses"],
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "TimesFMForecaster":
        checkpoint = os.path.join(path, "checkpoints", "final.pt")
        payload = torch.load(checkpoint, map_location="cpu")
        config = TimesFMConfig(**payload["config"])
        train_config = TorchTrainConfig(**payload["train_config"])
        instance = cls(config=config, pretrained_model=None, train_config=train_config)
        instance.model.load_state_dict(payload["state_dict"])
        if payload.get("scaler"):
            instance.scaler_ = StandardScaler1D(
                mean_=float(payload["scaler"]["mean"]),
                scale_=float(payload["scaler"]["scale"]),
                fitted_=True,
            )
        instance.losses_ = list(payload.get("losses", []))
        instance.backend_ = payload.get("backend", instance.backend_)
        instance.series_ = None
        return instance
