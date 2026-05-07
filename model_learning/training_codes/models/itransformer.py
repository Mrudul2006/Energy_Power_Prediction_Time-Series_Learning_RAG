"""
MODEL 13 - iTransformer.

iTransformer inverts the usual Transformer axes: each variable's full history
is embedded as a token, and attention operates across variables. This is most
useful for multivariate energy data with weather, region, or sensor channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import os

import numpy as np
import pandas as pd

from .common import (
    ForecastResult,
    StandardScaler1D,
    StandardScaler2D,
    calendar_features,
    ensure_dir,
    ensure_series,
    future_index,
    save_json,
)
from .neural_common import TorchTrainConfig, require_torch

torch, nn, F, DataLoader, TensorDataset = require_torch()


@dataclass
class iTransformerConfig:
    input_size: int = 336
    horizon: int = 24
    num_variables: int = 4
    d_model: int = 128
    n_heads: int = 8
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.10
    target_variable: int = 0


class iTransformerForecastNet(nn.Module):
    """Variable-token Transformer for multivariate forecasting."""

    def __init__(self, config: iTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.value_embedding = nn.Linear(config.input_size, config.d_model)
        self.variable_position = nn.Parameter(torch.zeros(1, config.num_variables, config.d_model))
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
        self.head = nn.Linear(config.d_model, config.horizon)

    def forward(self, x):
        """x shape: [batch, input_size, num_variables]."""

        if x.shape[1] != self.config.input_size:
            raise ValueError("input sequence length does not match config.input_size")
        z = x.transpose(1, 2)
        tokens = self.value_embedding(z) + self.variable_position[:, : z.shape[1], :]
        encoded = self.encoder(tokens)
        target = min(self.config.target_variable, encoded.shape[1] - 1)
        return self.head(encoded[:, target, :])


class iTransformerForecaster:
    """Trainable iTransformer wrapper with time-based features."""

    def __init__(self, model_config: iTransformerConfig | None = None, train_config: TorchTrainConfig | None = None) -> None:
        self.model_config = model_config or iTransformerConfig()
        self.train_config = train_config or TorchTrainConfig(
            input_size=self.model_config.input_size,
            horizon=self.model_config.horizon,
        )
        self.model = iTransformerForecastNet(self.model_config)
        self.scaler_ = None
        self.feature_scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None
        self.feature_columns_: list[str] = []

    def _build_features(self, index: pd.Index) -> pd.DataFrame:
        return calendar_features(index, include_india_holidays=True)

    def fit(
        self,
        data: pd.Series | pd.DataFrame | Sequence[float],
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        features: Optional[pd.DataFrame] = None,
    ) -> "iTransformerForecaster":
        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        self.series_ = y
        feature_frame = features if features is not None else self._build_features(y.index)
        feature_frame = feature_frame.reindex(y.index)
        feature_frame = feature_frame.fillna(method="ffill").fillna(0.0)
        self.feature_columns_ = feature_frame.columns.tolist()

        values = y.to_numpy(dtype=float)
        self.scaler_ = StandardScaler1D().fit(values)
        scaled = self.scaler_.transform(values)

        feature_values = feature_frame.to_numpy(dtype=float)
        self.feature_scaler_ = StandardScaler2D().fit(feature_values)
        feature_scaled = self.feature_scaler_.transform(feature_values)

        num_variables = 1 + feature_scaled.shape[1]
        if self.model_config.num_variables != num_variables:
            self.model_config.num_variables = num_variables
            self.model = iTransformerForecastNet(self.model_config)

        input_size = int(self.train_config.input_size)
        horizon = int(self.train_config.horizon)
        end = len(scaled) - input_size - horizon + 1
        if end <= 0:
            raise ValueError("not enough observations for iTransformer training windows")

        xs = []
        ys = []
        for start in range(0, end):
            past_y = scaled[start : start + input_size].reshape(-1, 1)
            past_feat = feature_scaled[start : start + input_size]
            x = np.concatenate([past_y, past_feat], axis=1)
            xs.append(x)
            ys.append(scaled[start + input_size : start + input_size + horizon])

        x_tensor = torch.tensor(np.asarray(xs), dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(ys), dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=True)
        device = torch.device(self.train_config.device)
        self.model.to(device)
        torch.manual_seed(int(self.train_config.random_state))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )

        self.losses_ = []
        for epoch in range(int(self.train_config.epochs)):
            self.model.train()
            total = 0.0
            count = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                if self.train_config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.train_config.gradient_clip))
                optimizer.step()
                total += float(loss.detach().cpu()) * len(xb)
                count += len(xb)
            self.losses_.append(total / max(1, count))
            if self.train_config.verbose:
                print(f"epoch={epoch + 1} loss={self.losses_[-1]:.6f}")
        return self

    def forecast(self, horizon: Optional[int] = None) -> ForecastResult:
        if self.series_ is None or self.scaler_ is None or self.feature_scaler_ is None:
            raise RuntimeError("fit the forecaster before forecast")
        h = int(horizon or self.train_config.horizon)
        input_size = int(self.train_config.input_size)
        values = self.series_.to_numpy(dtype=float)
        if len(values) < input_size:
            raise ValueError("series is shorter than input_size")
        scaled = self.scaler_.transform(values)
        feature_frame = self._build_features(self.series_.index)
        feature_frame = feature_frame.reindex(self.series_.index)
        feature_scaled = self.feature_scaler_.transform(feature_frame.to_numpy(dtype=float))
        past_y = scaled[-input_size:].reshape(-1, 1)
        past_feat = feature_scaled[-input_size:]
        x = np.concatenate([past_y, past_feat], axis=1)
        x_tensor = torch.tensor(x.reshape(1, input_size, -1), dtype=torch.float32)

        device = next(self.model.parameters()).device
        x_tensor = x_tensor.to(device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).detach().cpu().numpy().reshape(-1)
        point = self.scaler_.inverse_transform(pred[:h])
        return ForecastResult(
            mean=point,
            index=future_index(self.series_.index, len(point)),
            metadata={"model": "iTransformer", "config": self.model_config.__dict__},
        )

    def save(self, path: str) -> None:
        ensure_dir(path)
        checkpoint_dir = os.path.join(path, "checkpoints")
        ensure_dir(checkpoint_dir)
        payload = {
            "state_dict": self.model.state_dict(),
            "model_config": self.model_config.__dict__,
            "train_config": self.train_config.__dict__,
            "scaler": {"mean": float(self.scaler_.mean_), "scale": float(self.scaler_.scale_)} if self.scaler_ else None,
            "feature_scaler": {
                "mean": self.feature_scaler_.mean_.tolist(),
                "scale": self.feature_scaler_.scale_.tolist(),
            }
            if self.feature_scaler_ is not None
            else None,
            "feature_columns": self.feature_columns_,
            "losses": list(self.losses_ or []),
        }
        torch.save(payload, os.path.join(checkpoint_dir, "final.pt"))
        save_json(
            {
                "model": "iTransformer",
                "model_config": payload["model_config"],
                "train_config": payload["train_config"],
                "feature_columns": self.feature_columns_,
                "losses": payload["losses"],
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "iTransformerForecaster":
        checkpoint = os.path.join(path, "checkpoints", "final.pt")
        payload = torch.load(checkpoint, map_location="cpu")
        model_config = iTransformerConfig(**payload["model_config"])
        train_config = TorchTrainConfig(**payload["train_config"])
        instance = cls(model_config=model_config, train_config=train_config)
        instance.model.load_state_dict(payload["state_dict"])
        if payload.get("scaler"):
            instance.scaler_ = StandardScaler1D(
                mean_=float(payload["scaler"]["mean"]),
                scale_=float(payload["scaler"]["scale"]),
                fitted_=True,
            )
        if payload.get("feature_scaler"):
            fs = payload["feature_scaler"]
            instance.feature_scaler_ = StandardScaler2D(
                mean_=np.asarray(fs["mean"], dtype=float),
                scale_=np.asarray(fs["scale"], dtype=float),
                fitted_=True,
            )
        instance.feature_columns_ = list(payload.get("feature_columns", []))
        instance.losses_ = list(payload.get("losses", []))
        instance.series_ = None
        return instance
