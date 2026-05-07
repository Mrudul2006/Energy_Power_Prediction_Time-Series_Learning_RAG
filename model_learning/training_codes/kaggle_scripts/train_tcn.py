import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/tcn"

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
    return np.asarray(xs)[..., None], np.asarray(ys)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TCNForecastNet(nn.Module):
    def __init__(self, input_dim: int = 1, channels: int = 96, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1)
        dilations = [1, 2, 4, 8, 16, 24, 48, 96, 168]
        self.blocks = nn.Sequential(
            *[TCNResidualBlock(channels, kernel_size, int(d), dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, HORIZON),
        )

    def forward(self, x):
        z = x.transpose(1, 2)
        z = self.input_projection(z)
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

    model = TCNForecastNet()
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
        {"state_dict": model.state_dict(), "mean": mean, "std": std},
        os.path.join(OUTPUT_DIR, "model.pt"),
    )

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
