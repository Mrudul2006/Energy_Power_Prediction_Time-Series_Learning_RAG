import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/mamba"

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


class SelectiveSSMBlock(nn.Module):
    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.05) -> None:
        super().__init__()
        self.inner_dim = d_model * expansion
        self.in_proj = nn.Linear(d_model, self.inner_dim)
        self.delta_proj = nn.Linear(d_model, self.inner_dim)
        self.b_proj = nn.Linear(d_model, self.inner_dim)
        self.c_proj = nn.Linear(d_model, self.inner_dim)
        self.a_log = nn.Parameter(torch.zeros(self.inner_dim))
        self.d_skip = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, length, _ = x.shape
        u = torch.tanh(self.in_proj(x))
        delta = torch.nn.functional.softplus(self.delta_proj(x)) + 1e-4
        b_t = torch.tanh(self.b_proj(x))
        c_t = torch.tanh(self.c_proj(x))
        a = -torch.exp(self.a_log).view(1, -1)
        state = x.new_zeros((bsz, self.inner_dim))
        outputs = []
        for t in range(length):
            decay = torch.exp(delta[:, t, :] * a)
            state = decay * state + delta[:, t, :] * b_t[:, t, :] * u[:, t, :]
            y_t = c_t[:, t, :] * state + self.d_skip.view(1, -1) * u[:, t, :]
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        return self.norm(x + self.dropout(self.out_proj(y)))


class MambaForecastNet(nn.Module):
    def __init__(self, input_dim: int = 1, d_model: int = 96, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.blocks = nn.Sequential(*[SelectiveSSMBlock(d_model) for _ in range(num_layers)])
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

    model = MambaForecastNet()
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
