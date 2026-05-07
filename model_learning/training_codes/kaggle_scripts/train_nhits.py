import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/nhits"

INPUT_SIZE = 720
HORIZON = 168
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


def standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values)) if float(np.std(values)) > 1e-12 else 1.0
    return (values - mean) / std, mean, std


def make_windows(values: np.ndarray, input_size: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    end = len(values) - input_size - horizon + 1
    for start in range(0, end):
        xs.append(values[start : start + input_size])
        ys.append(values[start + input_size : start + input_size + horizon])
    return np.asarray(xs)[..., None], np.asarray(ys)


class NHITSBlock(nn.Module):
    def __init__(self, input_size: int, horizon: int, pooling_rate: int, interpolation_size: int, hidden_dim: int = 256):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.pooling_rate = int(pooling_rate)
        self.interpolation_size = int(interpolation_size)
        pooled_size = max(1, input_size // self.pooling_rate)
        self.mlp = nn.Sequential(
            nn.Linear(pooled_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.backcast_head = nn.Linear(hidden_dim, input_size)
        self.forecast_knots = nn.Linear(hidden_dim, self.interpolation_size)

    def _pool(self, x):
        if self.pooling_rate <= 1:
            return x
        pooled = torch.nn.functional.avg_pool1d(x.unsqueeze(1), kernel_size=self.pooling_rate, stride=self.pooling_rate)
        return pooled.squeeze(1)

    def _interpolate(self, knots):
        z = knots.unsqueeze(1)
        out = torch.nn.functional.interpolate(z, size=self.horizon, mode="linear", align_corners=False)
        return out.squeeze(1)

    def forward(self, x):
        pooled = self._pool(x)
        z = self.mlp(pooled)
        backcast = self.backcast_head(z)
        forecast = self._interpolate(self.forecast_knots(z))
        return backcast, forecast


class NHITSForecastNet(nn.Module):
    def __init__(self, input_size: int, horizon: int):
        super().__init__()
        rates = [1, 2, 4, 8]
        sizes = [168, 84, 42, 21]
        blocks = []
        for rate, size in zip(rates, sizes):
            blocks.append(NHITSBlock(input_size, horizon, rate, size))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        residual = x.squeeze(-1)
        forecast = residual.new_zeros((residual.shape[0], HORIZON))
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    y = load_series().to_numpy(dtype=float)
    scaled, mean, std = standardize(y)
    X, Y = make_windows(scaled, INPUT_SIZE, HORIZON)

    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=True)

    model = NHITSForecastNet(INPUT_SIZE, HORIZON)
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

    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std}, os.path.join(OUTPUT_DIR, "model.pt"))

    model.eval()
    with torch.no_grad():
        x_last = torch.tensor(scaled[-INPUT_SIZE:].reshape(1, INPUT_SIZE, 1), dtype=torch.float32)
        pred_scaled = model(x_last).numpy().reshape(-1)
    pred = pred_scaled * std + mean

    start = load_series().index[-1] + pd.Timedelta(hours=1)
    idx = pd.date_range(start=start, periods=HORIZON, freq="H")
    pd.DataFrame({"mean": pred}, index=idx).to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
