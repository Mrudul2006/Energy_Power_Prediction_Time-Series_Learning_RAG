"""
Small PyTorch utilities shared by neural forecasting modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import importlib
import os

import numpy as np

from .common import ForecastResult, StandardScaler1D, ensure_series, future_index, make_windows


def require_torch():
    """Import torch lazily so statistical modules do not require it."""

    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, F, DataLoader, TensorDataset


@dataclass
class TorchTrainConfig:
    input_size: int = 168
    horizon: int = 24
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.0
    device: str = "cpu"
    random_state: int = 7
    gradient_clip: Optional[float] = 1.0
    verbose: bool = False


def train_torch_forecaster(
    model,
    values: Sequence[float],
    config: TorchTrainConfig,
    n_features: int = 1,
) -> Tuple[object, StandardScaler1D, list[float]]:
    """Fit a torch model on univariate rolling windows."""

    torch, nn, F, DataLoader, TensorDataset = require_torch()
    torch.manual_seed(int(config.random_state))
    scaler = StandardScaler1D().fit(values)
    scaled = scaler.transform(values)
    X, y = make_windows(scaled, config.input_size, config.horizon)
    if n_features > 1:
        raise ValueError("train_torch_forecaster currently expects univariate windows")
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=config.batch_size, shuffle=True)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    losses: list[float] = []
    for epoch in range(int(config.epochs)):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        losses.append(total / max(1, count))
        if config.verbose:
            print(f"epoch={epoch + 1} loss={losses[-1]:.6f}")
    return model, scaler, losses


def forecast_torch_univariate(model, history: Sequence[float], scaler: StandardScaler1D, config: TorchTrainConfig):
    """Forecast from the latest input window of a fitted torch model."""

    torch, *_ = require_torch()
    if len(history) < config.input_size:
        raise ValueError("history is shorter than input_size")
    device = next(model.parameters()).device
    scaled = scaler.transform(history)
    x = torch.tensor(scaled[-config.input_size :].reshape(1, config.input_size, 1), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x).detach().cpu().numpy().reshape(-1)
    return scaler.inverse_transform(pred_scaled)


class TorchForecasterMixin:
    """Reusable fit/forecast mixin for univariate PyTorch models."""

    train_config: TorchTrainConfig
    model: object
    scaler_: Optional[StandardScaler1D]
    losses_: list[float]
    series_: object

    def fit(self, data, target_col=None, timestamp_col=None):
        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        self.series_ = y
        fitted, scaler, losses = train_torch_forecaster(self.model, y.to_numpy(dtype=float), self.train_config)
        self.model = fitted
        self.scaler_ = scaler
        self.losses_ = losses
        return self

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        if self.scaler_ is None or self.series_ is None:
            raise RuntimeError("fit the forecaster before forecast")
        if horizon is not None and int(horizon) != int(self.train_config.horizon):
            raise ValueError("this direct neural forecaster was trained for a fixed horizon")
        pred = forecast_torch_univariate(
            self.model,
            self.series_.to_numpy(dtype=float),
            self.scaler_,
            self.train_config,
        )
        return ForecastResult(
            mean=pred,
            index=future_index(self.series_.index, self.train_config.horizon),
            metadata={"training_loss": self.losses_[-1] if self.losses_ else None},
        )

    def save(self, path: str) -> None:
        from .common import ensure_dir, save_json

        torch, *_ = require_torch()
        ensure_dir(path)
        checkpoint_dir = os.path.join(path, "checkpoints")
        ensure_dir(checkpoint_dir)
        payload: Dict[str, Any] = {
            "state_dict": self.model.state_dict(),
            "model_config": getattr(self, "model_config", None).__dict__ if hasattr(self, "model_config") else None,
            "train_config": getattr(self, "train_config", None).__dict__ if hasattr(self, "train_config") else None,
            "model_config_class": (
                getattr(self, "model_config", None).__class__.__module__,
                getattr(self, "model_config", None).__class__.__name__,
            )
            if hasattr(self, "model_config")
            else None,
            "train_config_class": (
                getattr(self, "train_config", None).__class__.__module__,
                getattr(self, "train_config", None).__class__.__name__,
            )
            if hasattr(self, "train_config")
            else None,
            "scaler": {"mean": float(self.scaler_.mean_), "scale": float(self.scaler_.scale_)} if self.scaler_ else None,
            "losses": list(self.losses_ or []),
        }
        torch.save(payload, os.path.join(checkpoint_dir, "final.pt"))
        save_json(
            {
                "model": self.__class__.__name__,
                "model_config": payload["model_config"],
                "train_config": payload["train_config"],
                "losses": payload["losses"],
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "TorchForecasterMixin":
        torch, *_ = require_torch()
        checkpoint = os.path.join(path, "checkpoints", "final.pt")
        payload = torch.load(checkpoint, map_location="cpu")
        model_config = None
        train_config = None
        if payload.get("model_config") is not None and payload.get("model_config_class") is not None:
            module_name, class_name = payload["model_config_class"]
            model_cls = getattr(importlib.import_module(module_name), class_name)
            model_config = model_cls(**payload["model_config"])
        if payload.get("train_config") is not None and payload.get("train_config_class") is not None:
            module_name, class_name = payload["train_config_class"]
            train_cls = getattr(importlib.import_module(module_name), class_name)
            train_config = train_cls(**payload["train_config"])
        instance = cls(model_config=model_config, train_config=train_config)
        instance.model.load_state_dict(payload["state_dict"])
        if payload.get("scaler"):
            instance.scaler_ = StandardScaler1D(
                mean_=float(payload["scaler"]["mean"]),
                scale_=float(payload["scaler"]["scale"]),
                fitted_=True,
            )
        instance.losses_ = list(payload.get("losses", []))
        instance.series_ = None
        return instance
