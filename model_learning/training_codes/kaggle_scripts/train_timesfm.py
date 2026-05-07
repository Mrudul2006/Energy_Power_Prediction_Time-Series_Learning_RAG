import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/timesfm"

CONTEXT_LEN = 512
HORIZON = 128
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


class LocalTimesFMNet(nn.Module):
    def __init__(self, context_len: int, horizon_len: int, patch_len: int = 32, stride: int = 32, d_model: int = 128, n_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = 1 + (context_len - patch_len) // stride
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.n_patches * d_model),
            nn.Linear(self.n_patches * d_model, horizon_len),
        )

    def forward(self, x):
        z = x.squeeze(-1)
        patches = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches[:, : self.n_patches, :]
        tokens = self.patch_projection(patches) + self.position
        encoded = self.encoder(tokens)
        return self.head(encoded)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    y = load_series().to_numpy(dtype=float)
    scaled, mean, std = standardize(y)

    X, Y = make_windows(scaled, CONTEXT_LEN, HORIZON)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=True)

    model = LocalTimesFMNet(CONTEXT_LEN, HORIZON)
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
        x_last = torch.tensor(scaled[-CONTEXT_LEN:].reshape(1, CONTEXT_LEN, 1), dtype=torch.float32)
        pred_scaled = model(x_last).numpy().reshape(-1)
    pred = pred_scaled * std + mean

    start = load_series().index[-1] + pd.Timedelta(hours=1)
    idx = pd.date_range(start=start, periods=HORIZON, freq="H")
    pd.DataFrame({"mean": pred}, index=idx).to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
