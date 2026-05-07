import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/tft"

INPUT_SIZE = 168
HORIZON = 24
EPOCHS = 5
BATCH_SIZE = 64


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_series() -> pd.Series:
    df = pd.read_excel(DATA_PATH)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL, TARGET_COL]).sort_values(TIMESTAMP_COL)
    series = df.set_index(TIMESTAMP_COL)[TARGET_COL].astype(float)
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def calendar_features(index: pd.Index) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    frame["hour"] = index.hour.astype(float)
    frame["dayofweek"] = index.dayofweek.astype(float)
    frame["month"] = index.month.astype(float)
    frame["dayofyear"] = index.dayofyear.astype(float)
    frame["is_weekend"] = (index.dayofweek >= 5).astype(float)
    fixed = {(1, 26), (8, 15), (10, 2), (12, 25)}
    frame["is_india_fixed_holiday"] = [float((ts.month, ts.day) in fixed) for ts in index]
    return frame


def standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values)) if float(np.std(values)) > 1e-12 else 1.0
    return (values - mean) / std, mean, std


def standardize_2d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (values - mean) / std, mean, std


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(output_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        z = torch.nn.functional.elu(self.fc1(x))
        z = self.fc2(z)
        z = self.dropout(z)
        z = self.glu(z)
        return self.norm(self.skip(x) + z)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
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
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_layers = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)])
        self.k_layers = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)])
        self.v_layer = nn.Linear(hidden_dim, self.head_dim)
        self.out = nn.Linear(self.head_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shared_v = self.v_layer(x)
        contexts = []
        for q_layer, k_layer in zip(self.q_layers, self.k_layers):
            q = q_layer(x)
            k = k_layer(x)
            score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
            attn = torch.softmax(score, dim=-1)
            attn = self.dropout(attn)
            contexts.append(torch.matmul(attn, shared_v))
        context = torch.stack(contexts, dim=0).mean(dim=0)
        return self.out(context)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, past_input_dim: int, future_input_dim: int, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.past_vsn = VariableSelectionNetwork(past_input_dim, hidden_dim, dropout)
        self.future_vsn = VariableSelectionNetwork(future_input_dim, hidden_dim, dropout)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        self.static_enrichment = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.attention = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.positionwise = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, past_observed, future_known):
        past_selected, _ = self.past_vsn(past_observed)
        future_selected, _ = self.future_vsn(future_known)
        enc_out, state = self.encoder(past_selected)
        dec_out, _ = self.decoder(future_selected, state)
        temporal = torch.cat([enc_out, dec_out], dim=1)
        temporal = self.lstm_norm(temporal + self.lstm_gate(temporal))
        enriched = self.static_enrichment(temporal)
        attended = self.attention(enriched)
        processed = self.positionwise(attended + enriched)
        future_processed = processed[:, -HORIZON:, :]
        return self.output(future_processed).squeeze(-1)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    y = load_series()
    features = calendar_features(y.index).ffill().fillna(0.0)

    y_values = y.to_numpy(dtype=float)
    y_scaled, mean, std = standardize(y_values)

    feat_values = features.to_numpy(dtype=float)
    feat_scaled, feat_mean, feat_std = standardize_2d(feat_values)

    xs_past = []
    xs_future = []
    ys = []
    end = len(y_scaled) - INPUT_SIZE - HORIZON + 1
    for start in range(0, end):
        xs_past.append(y_scaled[start : start + INPUT_SIZE].reshape(-1, 1))
        xs_future.append(feat_scaled[start + INPUT_SIZE : start + INPUT_SIZE + HORIZON])
        ys.append(y_scaled[start + INPUT_SIZE : start + INPUT_SIZE + HORIZON])

    past_tensor = torch.tensor(np.asarray(xs_past), dtype=torch.float32)
    future_tensor = torch.tensor(np.asarray(xs_future), dtype=torch.float32)
    target_tensor = torch.tensor(np.asarray(ys), dtype=torch.float32)

    loader = DataLoader(TensorDataset(past_tensor, future_tensor, target_tensor), batch_size=BATCH_SIZE, shuffle=True)
    model = TemporalFusionTransformer(past_input_dim=1, future_input_dim=features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for _ in range(EPOCHS):
        model.train()
        for past_b, future_b, target_b in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(past_b, future_b)
            loss = torch.nn.functional.mse_loss(pred, target_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
        },
        os.path.join(OUTPUT_DIR, "model.pt"),
    )

    future_idx = pd.date_range(start=y.index[-1] + pd.Timedelta(hours=1), periods=HORIZON, freq="H")
    future_feat = calendar_features(future_idx).to_numpy(dtype=float)
    future_feat = (future_feat - feat_mean) / feat_std

    with torch.no_grad():
        past = torch.tensor(y_scaled[-INPUT_SIZE:].reshape(1, INPUT_SIZE, 1), dtype=torch.float32)
        future = torch.tensor(future_feat.reshape(1, HORIZON, -1), dtype=torch.float32)
        pred_scaled = model(past, future).numpy().reshape(-1)
    pred = pred_scaled * std + mean

    pd.DataFrame({"mean": pred}, index=future_idx).to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
