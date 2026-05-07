import os
import pickle

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/conformal"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_series(df: pd.DataFrame) -> pd.Series:
    frame = df.copy()
    frame[TIMESTAMP_COL] = pd.to_datetime(frame[TIMESTAMP_COL], errors="coerce")
    frame = frame.dropna(subset=[TIMESTAMP_COL, TARGET_COL]).sort_values(TIMESTAMP_COL)
    series = frame.set_index(TIMESTAMP_COL)[TARGET_COL].astype(float)
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def future_index(index: pd.Index, horizon: int) -> pd.Index:
    freq = index.freqstr or pd.infer_freq(index) or "H"
    start = index[-1] + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=start, periods=horizon, freq=freq)


def conformal_interval(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1) -> float:
    resid = np.abs(y_true - y_pred)
    q = np.quantile(resid, 1.0 - alpha)
    return float(q)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = pd.read_excel(DATA_PATH)
    y = ensure_series(df)

    split = int(len(y) * 0.8)
    train_y = y.iloc[:split]
    cal_y = y.iloc[split:]

    model = SARIMAX(
        train_y,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False, maxiter=300)

    pred = results.get_forecast(steps=len(cal_y)).predicted_mean.to_numpy(dtype=float)
    q = conformal_interval(cal_y.to_numpy(dtype=float), pred, alpha=0.1)

    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "results": results, "q": q}, f)

    horizon = 24
    mean = results.get_forecast(steps=horizon).predicted_mean.to_numpy(dtype=float)
    lower = mean - q
    upper = mean + q
    out = pd.DataFrame(
        {"mean": mean, "lower": lower, "upper": upper},
        index=future_index(y.index, horizon),
    )
    out.to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
