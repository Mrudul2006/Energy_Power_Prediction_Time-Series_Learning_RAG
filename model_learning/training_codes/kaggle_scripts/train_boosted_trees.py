import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
OUTPUT_DIR = "/kaggle/working/boosted_trees"



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
    return frame


def supervised_lag_frame(series: pd.Series, lags: list[int], rolling_windows: list[int], horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    y = series
    frame = pd.DataFrame(index=y.index)
    for lag in sorted(set(lags)):
        frame[f"lag_{lag}"] = y.shift(lag)
    for window in sorted(set(rolling_windows)):
        if window <= 1:
            continue
        shifted = y.shift(1)
        frame[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"roll_std_{window}"] = shifted.rolling(window).std()
    frame = frame.join(calendar_features(y.index))
    target = y.shift(-horizon)
    data = frame.join(target.rename("target")).dropna()
    return data.drop(columns=["target"]), data["target"]


def future_index(index: pd.Index, horizon: int) -> pd.Index:
    freq = index.freqstr or pd.infer_freq(index) or "H"
    start = index[-1] + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=start, periods=horizon, freq=freq)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = pd.read_excel(DATA_PATH)
    y = ensure_series(df)

    lags = [1, 2, 3, 24, 25, 48, 72, 168]
    rolls = [3, 6, 24, 168]
    horizon = 24

    X, target = supervised_lag_frame(y, lags, rolls, horizon=1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.035, max_leaf_nodes=63)
    model.fit(Xs, target)

    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "lags": lags, "rolls": rolls}, f)

    history = y.to_numpy(dtype=float).tolist()
    preds = []
    for ts in future_index(y.index, horizon):
        row = {}
        for lag in lags:
            row[f"lag_{lag}"] = history[-lag]
        for window in rolls:
            if window <= len(history):
                arr = np.asarray(history[-window:], dtype=float)
                row[f"roll_mean_{window}"] = float(np.mean(arr))
                row[f"roll_std_{window}"] = float(np.std(arr))
            else:
                row[f"roll_mean_{window}"] = np.nan
                row[f"roll_std_{window}"] = np.nan
        cal = calendar_features(pd.Index([ts]))
        for col in cal.columns:
            row[col] = float(cal[col].iloc[0])
        x_row = pd.DataFrame([row]).fillna(0.0)
        x_scaled = scaler.transform(x_row)
        pred = float(model.predict(x_scaled)[0])
        preds.append(pred)
        history.append(pred)

    out = pd.DataFrame({"mean": np.asarray(preds, dtype=float)}, index=future_index(y.index, horizon))
    out.to_csv(os.path.join(OUTPUT_DIR, "forecast.csv"), index=True)


if __name__ == "__main__":
    main()
