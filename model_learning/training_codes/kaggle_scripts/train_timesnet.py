import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/timesnet"

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


class Inception2D(nn.Module):
    def __init__(self, channels: int, kernel_sizes: list[int], dropout: float) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                    nn.Dropout2d(dropout),
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x):
        return torch.stack([conv(x) for conv in self.convs], dim=0).mean(dim=0)


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, kernel_sizes: list[int], dropout: float, periods: list[int]) -> None:
        super().__init__()
        self.conv = Inception2D(d_model, kernel_sizes, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.periods = periods

    def _period_conv(self, x, period: int):
        bsz, length, channels = x.shape
        pad_len = (period - length % period) % period
        x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_len)) if pad_len else x
        total = x_pad.shape[1]
        grid = x_pad.reshape(bsz, total // period, period, channels).permute(0, 3, 1, 2)
        out = self.conv(grid)
        out = out.permute(0, 2, 3, 1).reshape(bsz, total, channels)
        return out[:, :length, :]

    def forward(self, x):
        period_outputs = [self._period_conv(x, int(period)) for period in self.periods]
        z = torch.stack(period_outputs, dim=0).mean(dim=0)
        return self.norm(x + z)


class TimesNetForecastNet(nn.Module):
    def __init__(self, input_dim: int = 1, d_model: int = 64, periods: list[int] | None = None):
        super().__init__()
        periods = periods or [24, 168]
        self.embedding = nn.Linear(input_dim, d_model)
        self.blocks = nn.Sequential(*[TimesBlock(d_model, [1, 3, 5], 0.1, periods) for _ in range(3)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(INPUT_SIZE * d_model),
            nn.Linear(INPUT_SIZE * d_model, HORIZON),
        )

    def forward(self, x):
        z = self.embedding(x)
        z = self.blocks(z)
        return self.head(z)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    y = load_series().to_numpy(dtype=float)
    scaled, mean, std = standardize(y)
    X, Y = make_windows(scaled, INPUT_SIZE, HORIZON)

    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=True)

    model = TimesNetForecastNet()
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
