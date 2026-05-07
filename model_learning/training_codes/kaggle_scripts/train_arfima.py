import os
import pickle

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/arfima"


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


def fractional_diff_weights(d: float, max_lags: int = 4096, threshold: float = 1e-6) -> np.ndarray:
    weights = [1.0]
    for k in range(1, int(max_lags)):
        w = -weights[-1] * (float(d) - k + 1.0) / k
        weights.append(float(w))
        if k > 32 and abs(w) < threshold:
            break
    return np.asarray(weights, dtype=float)


def fractional_difference(values: np.ndarray, d: float, max_lags: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=float).reshape(-1)
    weights = fractional_diff_weights(d, max_lags=max_lags)
    z = np.convolve(y, weights, mode="full")[: len(y)]
    return z, weights


def inverse_fractional_forecast(history: np.ndarray, differenced_forecast: np.ndarray, weights: np.ndarray) -> np.ndarray:
    y_hist = list(np.asarray(history, dtype=float).reshape(-1))
    z_fore = np.asarray(differenced_forecast, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    out = []
    for z_t in z_fore:
        correction = 0.0
        max_k = min(len(w), len(y_hist) + 1)
        for k in range(1, max_k):
            correction += w[k] * y_hist[-k]
        y_t = float(z_t - correction)
        y_hist.append(y_t)
        out.append(y_t)
    return np.asarray(out, dtype=float)


def estimate_local_whittle_d(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float).reshape(-1)
    y = y[np.isfinite(y)] - np.nanmean(y)
    n = len(y)
    freqs, pxx = signal.periodogram(y)
    pxx = np.maximum(pxx[1:], 1e-18)
    m = max(20, int(n**0.65))
    m = min(int(m), len(pxx))
    lam = 2.0 * np.pi * np.arange(1, m + 1) / n
    periodogram = pxx[:m]

    def objective(d_value: float) -> float:
        adjusted = periodogram * np.power(lam, 2.0 * d_value)
        return float(np.log(np.mean(adjusted)) - 2.0 * d_value * np.mean(np.log(lam)))

    opt = optimize.minimize(lambda z: objective(float(z[0])), x0=[0.25], bounds=[(-0.45, 0.95)], method="L-BFGS-B")
    return float(opt.x[0])


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = pd.read_excel(DATA_PATH)
    y = ensure_series(df)

    d = estimate_local_whittle_d(y.to_numpy(dtype=float))
    z, weights = fractional_difference(y.to_numpy(dtype=float), d=d, max_lags=4096)
    z_series = pd.Series(z, index=y.index).replace([np.inf, -np.inf], np.nan).dropna()

    model = SARIMAX(
        z_series,
        order=(1, 0, 1),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False, maxiter=300)

    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "results": results, "d": d, "weights": weights, "series": y}, f)

    horizon = 24
    z_pred = results.get_forecast(steps=horizon).predicted_mean.to_numpy(dtype=float)
    y_pred = inverse_fractional_forecast(y.to_numpy(dtype=float), z_pred, weights)
    out = pd.DataFrame({"mean": y_pred}, index=future_index(y.index, horizon))
    out.to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
