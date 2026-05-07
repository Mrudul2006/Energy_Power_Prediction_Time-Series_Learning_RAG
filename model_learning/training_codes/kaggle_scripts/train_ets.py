import os
import pickle

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/ets"


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


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = pd.read_excel(DATA_PATH)
    y = ensure_series(df)

    model = ETSModel(
        y,
        error="add",
        trend="add",
        damped_trend=False,
        seasonal="add",
        seasonal_periods=24,
        initialization_method="estimated",
    )
    results = model.fit()

    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "results": results}, f)

    horizon = 24
    mean = results.forecast(steps=horizon).to_numpy(dtype=float)
    out = pd.DataFrame({"mean": mean}, index=future_index(y.index, horizon))
    out.to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
