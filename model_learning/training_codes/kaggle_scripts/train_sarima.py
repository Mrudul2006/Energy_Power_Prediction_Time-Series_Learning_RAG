import os
import pickle

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/sarima"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_series(df: pd.DataFrame) -> pd.Series:
    frame = df.copy()
    frame[TIMESTAMP_COL] = pd.to_datetime(frame[TIMESTAMP_COL], errors="coerce")
    frame = frame.dropna(subset=[TIMESTAMP_COL, TARGET_COL]).sort_values(TIMESTAMP_COL)
    series = frame.set_index(TIMESTAMP_COL)[TARGET_COL].astype(float)
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


def future_index(index: pd.Index, horizon: int) -> pd.Index:
    freq = index.freqstr or pd.infer_freq(index) or "H"
    start = index[-1] + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=start, periods=horizon, freq=freq)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = pd.read_excel(DATA_PATH)
    y = ensure_series(df)
    exog = calendar_features(y.index).ffill().fillna(0.0)

    model = SARIMAX(
        y,
        exog=exog,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False, maxiter=300)

    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "results": results}, f)

    horizon = 24
    future_idx = future_index(y.index, horizon)
    future_exog = calendar_features(future_idx).ffill().fillna(0.0)
    forecast = results.get_forecast(steps=horizon, exog=future_exog)
    mean = forecast.predicted_mean.to_numpy(dtype=float)
    conf = forecast.conf_int()
    out = pd.DataFrame(
        {
            "mean": mean,
            "lower": conf.iloc[:, 0].to_numpy(dtype=float),
            "upper": conf.iloc[:, 1].to_numpy(dtype=float),
        },
        index=future_idx,
    )
    out.to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
