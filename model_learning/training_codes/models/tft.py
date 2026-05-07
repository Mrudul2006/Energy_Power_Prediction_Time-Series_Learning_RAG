"""
MODEL 11 - Temporal Fusion Transformer (TFT).

This is a compact but faithful implementation of TFT's main ingredients:
variable selection networks, gated residual networks, LSTM encoder/decoder,
interpretable multi-head attention, and quantile output heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
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
class TFTConfig:
    past_input_dim: int
    future_input_dim: int
    horizon: int = 24
    hidden_dim: int = 128
    lstm_layers: int = 1
    attention_heads: int = 4
    dropout: float = 0.10
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """Nonlinear transformation with optional context, gating, and skip path."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None, context_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim is not None else None
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(output_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, context=None):
        z = self.fc1(x)
        if context is not None and self.context is not None:
            while context.dim() < z.dim():
                context = context.unsqueeze(1)
            z = z + self.context(context)
        z = F.elu(z)
        z = self.fc2(z)
        z = self.dropout(z)
        z = self.glu(z)
        return self.norm(self.skip(x) + z)


class VariableSelectionNetwork(nn.Module):
    """
    Softly select scalar input variables and transform them into hidden_dim.

    Input shape: [batch, time, variables].
    Output shape: [batch, time, hidden_dim], weights: [batch, time, variables].
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.weight_grn = GatedResidualNetwork(input_dim, hidden_dim, input_dim, dropout=dropout)
        self.var_grns = nn.ModuleList(
            [GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout=dropout) for _ in range(input_dim)]
        )

    def forward(self, x):
        weights = torch.softmax(self.weight_grn(x), dim=-1)
        transformed = torch.stack([grn(x[..., i : i + 1]) for i, grn in enumerate(self.var_grns)], dim=-2)
        selected = torch.sum(weights.unsqueeze(-1) * transformed, dim=-2)
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with a shared value projection, matching TFT's
    interpretable attention design.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_layers = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)])
        self.k_layers = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)])
        self.v_layer = nn.Linear(hidden_dim, self.head_dim)
        self.out = nn.Linear(self.head_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None) -> Tuple[object, object]:
        shared_v = self.v_layer(x)
        contexts = []
        weights = []
        for q_layer, k_layer in zip(self.q_layers, self.k_layers):
            q = q_layer(x)
            k = k_layer(x)
            score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
            if mask is not None:
                score = score.masked_fill(mask == 0, -1e9)
            attn = torch.softmax(score, dim=-1)
            attn = self.dropout(attn)
            contexts.append(torch.matmul(attn, shared_v))
            weights.append(attn)
        context = torch.stack(contexts, dim=0).mean(dim=0)
        attention_weights = torch.stack(weights, dim=1)
        return self.out(context), attention_weights


class TemporalFusionTransformer(nn.Module):
    """
    TFT forward pass.

    past_observed: [batch, encoder_length, past_input_dim]
    future_known: [batch, horizon, future_input_dim]
    returns: [batch, horizon, n_quantiles]
    """

    def __init__(self, config: TFTConfig) -> None:
        super().__init__()
        self.config = config
        self.past_vsn = VariableSelectionNetwork(config.past_input_dim, config.hidden_dim, config.dropout)
        self.future_vsn = VariableSelectionNetwork(config.future_input_dim, config.hidden_dim, config.dropout)
        self.encoder = nn.LSTM(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.lstm_gate = GatedLinearUnit(config.hidden_dim, config.hidden_dim)
        self.lstm_norm = nn.LayerNorm(config.hidden_dim)
        self.static_enrichment = GatedResidualNetwork(config.hidden_dim, config.hidden_dim, dropout=config.dropout)
        self.attention = InterpretableMultiHeadAttention(config.hidden_dim, config.attention_heads, config.dropout)
        self.positionwise = GatedResidualNetwork(config.hidden_dim, config.hidden_dim, dropout=config.dropout)
        self.output = nn.Linear(config.hidden_dim, len(config.quantiles))
        self.last_attention_weights = None
        self.last_past_selection = None
        self.last_future_selection = None

    def forward(self, past_observed, future_known):
        past_selected, past_weights = self.past_vsn(past_observed)
        future_selected, future_weights = self.future_vsn(future_known)
        enc_out, state = self.encoder(past_selected)
        dec_out, _ = self.decoder(future_selected, state)
        temporal = torch.cat([enc_out, dec_out], dim=1)
        temporal = self.lstm_norm(temporal + self.lstm_gate(temporal))
        enriched = self.static_enrichment(temporal)
        attended, attn = self.attention(enriched)
        processed = self.positionwise(attended + enriched)
        future_processed = processed[:, -self.config.horizon :, :]
        raw = self.output(future_processed)
        if len(self.config.quantiles) > 1:
            raw, _ = torch.sort(raw, dim=-1)
        self.last_attention_weights = attn
        self.last_past_selection = past_weights
        self.last_future_selection = future_weights
        return raw


def quantile_loss(y_true, y_pred, quantiles: Sequence[float]):
    """TFT-style summed pinball loss for [batch,horizon,n_quantiles]."""

    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(-1)
    losses = []
    for i, q in enumerate(quantiles):
        err = y_true[..., 0] - y_pred[..., i]
        losses.append(torch.maximum((q - 1.0) * err, q * err).unsqueeze(-1))
    return torch.mean(torch.cat(losses, dim=-1))


class TFTForecaster:
    """Trainable TFT wrapper using time-based future-known features."""

    def __init__(self, model_config: TFTConfig, train_config: TorchTrainConfig | None = None) -> None:
        self.model_config = model_config
        self.train_config = train_config or TorchTrainConfig(
            input_size=168,
            horizon=model_config.horizon,
        )
        self.model = TemporalFusionTransformer(self.model_config)
        self.scaler_ = None
        self.feature_scaler_ = None
        self.losses_: list[float] = []
        self.series_ = None
        self.feature_columns_: list[str] = []

    def _build_features(self, index: pd.Index) -> pd.DataFrame:
        frame = calendar_features(index, include_india_holidays=True)
        return frame

    def fit(
        self,
        data: pd.Series | pd.DataFrame | Sequence[float],
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        future_features: Optional[pd.DataFrame] = None,
    ) -> "TFTForecaster":
        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        self.series_ = y
        feature_frame = future_features if future_features is not None else self._build_features(y.index)
        feature_frame = feature_frame.reindex(y.index)
        feature_frame = feature_frame.fillna(method="ffill").fillna(0.0)
        self.feature_columns_ = feature_frame.columns.tolist()

        values = y.to_numpy(dtype=float)
        self.scaler_ = StandardScaler1D().fit(values)
        scaled = self.scaler_.transform(values)

        feature_values = feature_frame.to_numpy(dtype=float)
        self.feature_scaler_ = StandardScaler2D().fit(feature_values)
        feature_scaled = self.feature_scaler_.transform(feature_values)

        input_size = int(self.train_config.input_size)
        horizon = int(self.train_config.horizon)
        end = len(scaled) - input_size - horizon + 1
        if end <= 0:
            raise ValueError("not enough observations for TFT training windows")

        past_list = []
        future_list = []
        target_list = []
        for start in range(0, end):
            past_list.append(scaled[start : start + input_size].reshape(-1, 1))
            future_list.append(feature_scaled[start + input_size : start + input_size + horizon])
            target_list.append(scaled[start + input_size : start + input_size + horizon])

        past = torch.tensor(np.asarray(past_list), dtype=torch.float32)
        future = torch.tensor(np.asarray(future_list), dtype=torch.float32)
        target = torch.tensor(np.asarray(target_list), dtype=torch.float32)

        dataset = TensorDataset(past, future, target)
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
            for past_b, future_b, target_b in loader:
                past_b = past_b.to(device)
                future_b = future_b.to(device)
                target_b = target_b.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(past_b, future_b)
                loss = quantile_loss(target_b, pred, self.model_config.quantiles)
                loss.backward()
                if self.train_config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.train_config.gradient_clip))
                optimizer.step()
                total += float(loss.detach().cpu()) * len(past_b)
                count += len(past_b)
            self.losses_.append(total / max(1, count))
            if self.train_config.verbose:
                print(f"epoch={epoch + 1} loss={self.losses_[-1]:.6f}")
        return self

    def forecast(self, horizon: Optional[int] = None, future_features: Optional[pd.DataFrame] = None) -> ForecastResult:
        if self.series_ is None or self.scaler_ is None or self.feature_scaler_ is None:
            raise RuntimeError("fit the forecaster before forecast")
        h = int(horizon or self.train_config.horizon)
        input_size = int(self.train_config.input_size)
        values = self.series_.to_numpy(dtype=float)
        if len(values) < input_size:
            raise ValueError("series is shorter than input_size")
        scaled = self.scaler_.transform(values)
        past = torch.tensor(scaled[-input_size:].reshape(1, input_size, 1), dtype=torch.float32)

        if future_features is None:
            idx = future_index(self.series_.index, h)
            future_features = self._build_features(idx)
        future_features = future_features.copy()
        future_features = future_features.reindex(future_features.index)
        future_scaled = self.feature_scaler_.transform(future_features.to_numpy(dtype=float))
        future = torch.tensor(future_scaled.reshape(1, h, -1), dtype=torch.float32)

        device = next(self.model.parameters()).device
        past = past.to(device)
        future = future.to(device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(past, future).detach().cpu().numpy().reshape(h, -1)
        point = self.scaler_.inverse_transform(pred[:, 0])
        return ForecastResult(
            mean=point,
            index=future_index(self.series_.index, len(point)),
            metadata={"model": "TFT", "config": self.model_config.__dict__},
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
                "model": "TFT",
                "model_config": payload["model_config"],
                "train_config": payload["train_config"],
                "feature_columns": self.feature_columns_,
                "losses": payload["losses"],
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "TFTForecaster":
        checkpoint = os.path.join(path, "checkpoints", "final.pt")
        payload = torch.load(checkpoint, map_location="cpu")
        model_config = TFTConfig(**payload["model_config"])
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
