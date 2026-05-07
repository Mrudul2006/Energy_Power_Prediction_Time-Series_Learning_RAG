import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/nbeats"

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
    return np.asarray(xs), np.asarray(ys)


class TrendBasis(nn.Module):
    def __init__(self, input_size: int, horizon: int, degree: int) -> None:
        super().__init__()
        backcast_t = torch.linspace(-1.0, 0.0, input_size)
        forecast_t = torch.linspace(0.0, 1.0, horizon)
        self.register_buffer("backcast_basis", torch.stack([backcast_t**i for i in range(degree + 1)]))
        self.register_buffer("forecast_basis", torch.stack([forecast_t**i for i in range(degree + 1)]))

    def forward(self, theta_b, theta_f):
        return theta_b @ self.backcast_basis, theta_f @ self.forecast_basis


class SeasonalityBasis(nn.Module):
    def __init__(self, input_size: int, horizon: int, harmonics: int) -> None:
        super().__init__()
        def build(length: int):
            t = torch.arange(length, dtype=torch.float32) / max(1, length)
            rows = []
            for k in range(1, harmonics + 1):
                rows.append(torch.sin(2.0 * np.pi * k * t))
                rows.append(torch.cos(2.0 * np.pi * k * t))
            return torch.stack(rows)
        self.register_buffer("backcast_basis", build(input_size))
        self.register_buffer("forecast_basis", build(horizon))

    def forward(self, theta_b, theta_f):
        return theta_b @ self.backcast_basis, theta_f @ self.forecast_basis


class NBEATSBlock(nn.Module):
    def __init__(self, input_size: int, horizon: int, block_type: str, hidden_dim: int = 256, layers: int = 4, degree: int = 3, harmonics: int = 12):
        super().__init__()
        self.block_type = block_type
        mlp = []
        in_dim = input_size
        for _ in range(layers):
            mlp.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp)
        if block_type == "trend":
            theta_dim = degree + 1
            self.basis = TrendBasis(input_size, horizon, degree)
            self.theta = nn.Linear(hidden_dim, theta_dim * 2)
            self.backcast_linear = None
            self.forecast_linear = None
        elif block_type == "seasonality":
            theta_dim = harmonics * 2
            self.basis = SeasonalityBasis(input_size, horizon, harmonics)
            self.theta = nn.Linear(hidden_dim, theta_dim * 2)
            self.backcast_linear = None
            self.forecast_linear = None
        else:
            self.basis = None
            self.theta = nn.Linear(hidden_dim, hidden_dim)
            self.backcast_linear = nn.Linear(hidden_dim, input_size)
            self.forecast_linear = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        z = self.mlp(x)
        theta = self.theta(z)
        if self.block_type == "generic":
            return self.backcast_linear(theta), self.forecast_linear(theta)
        theta_b, theta_f = torch.chunk(theta, 2, dim=-1)
        return self.basis(theta_b, theta_f)


class NBEATSForecastNet(nn.Module):
    def __init__(self, input_size: int, horizon: int):
        super().__init__()
        stack_types = ["trend", "seasonality", "generic", "generic"]
        blocks = []
        for stack in stack_types:
            for _ in range(2):
                blocks.append(NBEATSBlock(input_size, horizon, stack))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        residual = x
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

    model = NBEATSForecastNet(INPUT_SIZE, HORIZON)
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
        x_last = torch.tensor(scaled[-INPUT_SIZE:].reshape(1, INPUT_SIZE), dtype=torch.float32)
        pred_scaled = model(x_last).numpy().reshape(-1)
    pred = pred_scaled * std + mean

    start = load_series().index[-1] + pd.Timedelta(hours=1)
    idx = pd.date_range(start=start, periods=HORIZON, freq="H")
    pd.DataFrame({"mean": pred}, index=idx).to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
