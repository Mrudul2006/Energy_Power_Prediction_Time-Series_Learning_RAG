"""
Shared utilities for the forecasting model implementations.

The helpers in this file deliberately avoid any heavy optional dependency.
They provide a consistent series coercion path, forecast result container,
calendar feature construction, lag-window construction, and lightweight
scaling used by both statistical and neural models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import importlib
import json
import math
import os
import pickle

import numpy as np
import pandas as pd


def require_package(package: str, install_hint: Optional[str] = None) -> Any:
    """Import an optional dependency or raise an actionable ImportError."""

    try:
        return importlib.import_module(package)
    except ImportError as exc:
        hint = install_hint or package
        raise ImportError(f"Optional dependency '{package}' is required. Install it with: pip install {hint}") from exc


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_series(
    data: pd.Series | pd.DataFrame | Sequence[float],
    target_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    name: str = "y",
) -> pd.Series:
    """
    Convert common user inputs into a numeric pandas Series.

    DataFrame inputs need either target_col or a single numeric column. If a
    timestamp column is supplied, it becomes a DatetimeIndex.
    """

    if isinstance(data, pd.Series):
        series = data.copy()
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
        if timestamp_col is not None:
            frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
            frame = frame.set_index(timestamp_col).sort_index()
        if target_col is None:
            numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) != 1:
                raise ValueError("target_col is required when the DataFrame has zero or multiple numeric columns")
            target_col = numeric_cols[0]
        series = frame[target_col].copy()
    else:
        series = pd.Series(np.asarray(data, dtype=float), name=name)

    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.RangeIndex(len(series))
    series = pd.to_numeric(series, errors="coerce").astype(float)
    series.name = series.name or name
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def infer_series_freq(index: pd.Index) -> Optional[str]:
    """Infer a pandas frequency string when the index is datetime-like."""

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    return index.freqstr or pd.infer_freq(index)


def future_index(index: pd.Index, horizon: int, freq: Optional[str] = None) -> pd.Index:
    """Build an index for the next horizon points."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if isinstance(index, pd.DatetimeIndex) and len(index) > 0:
        resolved_freq = freq or infer_series_freq(index) or "H"
        start = index[-1] + pd.tseries.frequencies.to_offset(str(resolved_freq).replace('H', 'h'))
        return pd.date_range(start=start, periods=horizon, freq=str(resolved_freq).replace('H', 'h'))
    start_value = int(index[-1]) + 1 if len(index) else 0
    return pd.RangeIndex(start_value, start_value + horizon)


@dataclass
class ForecastResult:
    """Unified result returned by model wrappers."""

    mean: np.ndarray
    index: Optional[pd.Index] = None
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    quantiles: Dict[float, np.ndarray] = field(default_factory=dict)
    components: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        if self.index is None:
            self.index = pd.RangeIndex(len(self.mean))
        if len(self.index) != len(self.mean):
            raise ValueError("index length must match forecast mean length")
        if self.lower is not None:
            self.lower = np.asarray(self.lower, dtype=float).reshape(-1)
        if self.upper is not None:
            self.upper = np.asarray(self.upper, dtype=float).reshape(-1)

    def to_frame(self) -> pd.DataFrame:
        """Return a tabular view with mean, intervals, quantiles, and components."""

        frame = pd.DataFrame({"mean": self.mean}, index=self.index)
        if self.lower is not None:
            frame["lower"] = self.lower
        if self.upper is not None:
            frame["upper"] = self.upper
        for q, values in sorted(self.quantiles.items()):
            frame[f"q{q:g}"] = np.asarray(values, dtype=float).reshape(-1)
        for name, values in self.components.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if len(arr) == len(frame):
                frame[name] = arr
        return frame


@dataclass
class StandardScaler1D:
    """Tiny scaler for univariate arrays with inverse transform support."""

    mean_: float = 0.0
    scale_: float = 1.0
    fitted_: bool = False

    def fit(self, values: Sequence[float]) -> "StandardScaler1D":
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("cannot fit scaler on empty/non-finite data")
        self.mean_ = float(np.mean(finite))
        scale = float(np.std(finite))
        self.scale_ = scale if scale > 1e-12 else 1.0
        self.fitted_ = True
        return self

    def transform(self, values: Sequence[float]) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("fit the scaler first")
        return (np.asarray(values, dtype=float) - self.mean_) / self.scale_

    def inverse_transform(self, values: Sequence[float]) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("fit the scaler first")
        return np.asarray(values, dtype=float) * self.scale_ + self.mean_


@dataclass
class StandardScaler2D:
    """Lightweight scaler for 2D feature arrays."""

    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    fitted_: bool = False

    def fit(self, values: np.ndarray) -> "StandardScaler2D":
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError("StandardScaler2D expects a 2D array")
        self.mean_ = np.nanmean(arr, axis=0)
        scale = np.nanstd(arr, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        self.scale_ = scale
        self.fitted_ = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.fitted_ or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit the scaler first")
        arr = np.asarray(values, dtype=float)
        return (arr - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if not self.fitted_ or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit the scaler first")
        arr = np.asarray(values, dtype=float)
        return arr * self.scale_ + self.mean_


def _time_position(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        delta = index - index[0]
        return delta.total_seconds().to_numpy(dtype=float) / 3600.0
    return np.arange(len(index), dtype=float)


def fourier_features(position: Sequence[float], period: float, order: int, prefix: str) -> pd.DataFrame:
    """Create Fourier sine/cosine columns for a numeric position vector."""

    pos = np.asarray(position, dtype=float)
    cols: Dict[str, np.ndarray] = {}
    for k in range(1, int(order) + 1):
        angle = 2.0 * math.pi * k * pos / float(period)
        cols[f"{prefix}_sin_{k}"] = np.sin(angle)
        cols[f"{prefix}_cos_{k}"] = np.cos(angle)
    return pd.DataFrame(cols)


def calendar_features(
    index: pd.Index,
    periods: Sequence[int] = (24, 168, 8760),
    orders: Mapping[int, int] | None = None,
    include_india_holidays: bool = True,
) -> pd.DataFrame:
    """
    Build deterministic future-known calendar features.

    The function works for DatetimeIndex and RangeIndex. For datetime data it
    includes hour, weekday, month, weekend, day-of-year and a compact India
    fixed-holiday indicator. For non-datetime data it uses elapsed integer time.
    """

    orders = dict(orders or {24: 4, 168: 3, 8760: 3})
    frame = pd.DataFrame(index=index)
    pos = _time_position(index)

    if isinstance(index, pd.DatetimeIndex):
        frame["hour"] = index.hour.astype(float)
        frame["dayofweek"] = index.dayofweek.astype(float)
        frame["month"] = index.month.astype(float)
        frame["dayofyear"] = index.dayofyear.astype(float)
        frame["is_weekend"] = (index.dayofweek >= 5).astype(float)
        if include_india_holidays:
            fixed = {
                (1, 26),   # Republic Day
                (8, 15),   # Independence Day
                (10, 2),   # Gandhi Jayanti
                (12, 25),  # Christmas, often a grid demand anomaly
            }
            frame["is_india_fixed_holiday"] = [float((ts.month, ts.day) in fixed) for ts in index]
    else:
        frame["time"] = pos

    for period in periods:
        order = int(orders.get(int(period), 1))
        fourier = fourier_features(pos, float(period), order, prefix=f"p{int(period)}")
        fourier.index = index
        frame = pd.concat([frame, fourier], axis=1)
    return frame.astype(float)


def make_windows(
    values: Sequence[float],
    input_size: int,
    horizon: int,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling supervised windows X=[n,input_size,1], y=[n,horizon]."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    if input_size <= 0 or horizon <= 0:
        raise ValueError("input_size and horizon must be positive")
    end = len(arr) - input_size - horizon + 1
    if end <= 0:
        raise ValueError("not enough observations for the requested input_size and horizon")
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for start in range(0, end, max(1, step)):
        xs.append(arr[start : start + input_size])
        ys.append(arr[start + input_size : start + input_size + horizon])
    return np.asarray(xs, dtype=float)[..., None], np.asarray(ys, dtype=float)


def supervised_lag_frame(
    series: pd.Series,
    lags: Sequence[int],
    rolling_windows: Sequence[int] = (),
    horizon: int = 1,
    calendar: bool = True,
    target_calendar: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a one-target lag-feature design matrix for y[t+horizon]."""

    y = ensure_series(series)
    frame = pd.DataFrame(index=y.index)
    for lag in sorted(set(int(lag) for lag in lags)):
        if lag <= 0:
            raise ValueError("lags must be positive")
        frame[f"lag_{lag}"] = y.shift(lag)
    for window in sorted(set(int(w) for w in rolling_windows)):
        if window <= 1:
            continue
        shifted = y.shift(1)
        frame[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"roll_std_{window}"] = shifted.rolling(window).std()
        frame[f"roll_min_{window}"] = shifted.rolling(window).min()
        frame[f"roll_max_{window}"] = shifted.rolling(window).max()
    h = int(horizon)
    if calendar:
        cal_index = y.index
        if target_calendar:
            if isinstance(y.index, pd.DatetimeIndex):
                freq = infer_series_freq(y.index) or "h"
                cal_index = y.index + h * pd.tseries.frequencies.to_offset(freq)
            else:
                cal_index = pd.RangeIndex(h, h + len(y))
        cal = calendar_features(cal_index)
        cal.index = y.index
        frame = pd.concat([frame, cal], axis=1)
    target = y.shift(-h).rename("target")
    combined = pd.concat([frame, target], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    return combined.drop(columns=["target"]), combined["target"]


def lag_feature_row(
    history: Sequence[float],
    timestamp: Any,
    lags: Sequence[int],
    rolling_windows: Sequence[int] = (),
    calendar_index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Build a single-row lag feature matrix for recursive/direct forecasting."""

    arr = np.asarray(history, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("history cannot be empty")
    row: Dict[str, float] = {}
    for lag in sorted(set(int(lag) for lag in lags)):
        row[f"lag_{lag}"] = float(arr[-lag]) if arr.size >= lag else float(arr[0])
    for window in sorted(set(int(w) for w in rolling_windows)):
        if window <= 1:
            continue
        window_values = arr[-window:] if arr.size >= window else arr
        row[f"roll_mean_{window}"] = float(np.mean(window_values))
        row[f"roll_std_{window}"] = float(np.std(window_values))
        row[f"roll_min_{window}"] = float(np.min(window_values))
        row[f"roll_max_{window}"] = float(np.max(window_values))

    if calendar_index is None:
        if isinstance(timestamp, pd.Timestamp):
            calendar_index = pd.DatetimeIndex([timestamp])
        else:
            calendar_index = pd.Index([timestamp])
    cal = calendar_features(calendar_index)
    for col, value in cal.iloc[0].items():
        row[col] = float(value)
    return pd.DataFrame([row])


def empirical_interval(
    point: Sequence[float],
    residuals: Sequence[float],
    alpha: float = 0.05,
    horizon_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a simple residual-quantile interval around point forecasts."""

    pred = np.asarray(point, dtype=float).reshape(-1)
    resid = np.asarray(residuals, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return pred.copy(), pred.copy()
    lower_q = float(np.quantile(resid, alpha / 2.0))
    upper_q = float(np.quantile(resid, 1.0 - alpha / 2.0))
    if horizon_scale:
        scale = np.sqrt(np.arange(1, len(pred) + 1, dtype=float))
    else:
        scale = np.ones_like(pred)
    return pred + lower_q * scale, pred + upper_q * scale


def pinball_loss_np(y_true: Sequence[float], y_pred: Sequence[float], quantile: float) -> float:
    """NumPy pinball loss for quantile forecasts."""

    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    diff = y - pred
    q = float(quantile)
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))
