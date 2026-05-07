import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/itransformer"

INPUT_SIZE = 336
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


class iTransformerForecastNet(nn.Module):
    def __init__(self, num_variables: int, d_model: int = 128, n_heads: int = 8, num_layers: int = 3):
        super().__init__()
        self.value_embedding = nn.Linear(INPUT_SIZE, d_model)
        self.variable_position = nn.Parameter(torch.zeros(1, num_variables, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, HORIZON)

    def forward(self, x):
        z = x.transpose(1, 2)
        tokens = self.value_embedding(z) + self.variable_position[:, : z.shape[1], :]
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0, :])


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    y = load_series()
    features = calendar_features(y.index).ffill().fillna(0.0)

    y_values = y.to_numpy(dtype=float)
    y_scaled, mean, std = standardize(y_values)

    feat_values = features.to_numpy(dtype=float)
    feat_scaled, feat_mean, feat_std = standardize_2d(feat_values)

    xs = []
    ys = []
    end = len(y_scaled) - INPUT_SIZE - HORIZON + 1
    for start in range(0, end):
        past_y = y_scaled[start : start + INPUT_SIZE].reshape(-1, 1)
        past_feat = feat_scaled[start : start + INPUT_SIZE]
        x = np.concatenate([past_y, past_feat], axis=1)
        xs.append(x)
        ys.append(y_scaled[start + INPUT_SIZE : start + INPUT_SIZE + HORIZON])

    x_tensor = torch.tensor(np.asarray(xs), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(ys), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=True)

    model = iTransformerForecastNet(num_variables=1 + features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
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

    past_y = y_scaled[-INPUT_SIZE:].reshape(-1, 1)
    past_feat = feat_scaled[-INPUT_SIZE:]
    x_last = np.concatenate([past_y, past_feat], axis=1)

    with torch.no_grad():
        pred_scaled = model(torch.tensor(x_last.reshape(1, INPUT_SIZE, -1), dtype=torch.float32)).numpy().reshape(-1)
    pred = pred_scaled * std + mean

    start = y.index[-1] + pd.Timedelta(hours=1)
    idx = pd.date_range(start=start, periods=HORIZON, freq="H")
    pd.DataFrame({"mean": pred}, index=idx).to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
