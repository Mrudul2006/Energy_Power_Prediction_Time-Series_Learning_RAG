import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/tbats"


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


def tbats_fourier_matrix(n: int, seasonal_periods: list[float], harmonics: dict[float, int], start: int = 0) -> pd.DataFrame:
    t = np.arange(start, start + int(n), dtype=float)
    cols = {}
    for period in seasonal_periods:
        period_f = float(period)
        h_count = int(harmonics.get(period, harmonics.get(period_f, 1)))
        for k in range(1, h_count + 1):
            angle = 2.0 * np.pi * k * t / period_f
            safe_period = str(period).replace(".", "_")
            cols[f"s{safe_period}_sin_{k}"] = np.sin(angle)
            cols[f"s{safe_period}_cos_{k}"] = np.cos(angle)
    return pd.DataFrame(cols)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = pd.read_excel(DATA_PATH)
    y = ensure_series(df)

    seasonal_periods = [24.0, 168.0, 8760.0]
    harmonics = {24.0: 10, 168.0: 8, 8760.0: 5}
    shift = float(max(0.0, 1.0 - np.nanmin(y.to_numpy(dtype=float))))
    lam = float(stats.boxcox_normmax(y.to_numpy(dtype=float) + shift, method="mle"))
    transformed = stats.boxcox(y.to_numpy(dtype=float) + shift, lmbda=lam)
    exog = tbats_fourier_matrix(len(y), seasonal_periods, harmonics, start=0)

    model = SARIMAX(
        transformed,
        exog=exog,
        order=(2, 0, 2),
        trend="ct",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False, maxiter=300)

    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "results": results, "lambda": lam, "shift": shift}, f)

    horizon = 24
    exog_future = tbats_fourier_matrix(horizon, seasonal_periods, harmonics, start=len(y))
    pred = results.get_forecast(steps=horizon, exog=exog_future)
    mean_t = np.asarray(pred.predicted_mean, dtype=float)
    mean = np.power(np.maximum(lam * mean_t + 1.0, 1e-12), 1.0 / lam) - shift
    out = pd.DataFrame({"mean": mean}, index=future_index(y.index, horizon))
    out.to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
