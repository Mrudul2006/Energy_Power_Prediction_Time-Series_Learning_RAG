"""
MODEL 2 - ARFIMA.

The implementation is an approximate ARFIMA(p,d,q):
1. Estimate or accept fractional integration d.
2. Apply fixed-width fractional differencing (1 - B)^d.
3. Fit an ARMA(p,q) to the fractionally differenced series.
4. Forecast the ARMA component and recursively invert fractional differencing.

This is the standard practical route when a full exact ARFIMA likelihood
implementation is not available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple
import os

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

from .common import (
    ForecastResult,
    empirical_interval,
    ensure_dir,
    ensure_series,
    future_index,
    load_pickle,
    require_package,
    save_json,
    save_pickle,
)


def fractional_diff_weights(d: float, max_lags: int = 4096, threshold: float = 1e-6) -> np.ndarray:
    """Coefficients of (1 - B)^d using the recursive binomial expansion."""

    weights = [1.0]
    for k in range(1, int(max_lags)):
        w = -weights[-1] * (float(d) - k + 1.0) / k
        weights.append(float(w))
        if k > 32 and abs(w) < threshold:
            break
    return np.asarray(weights, dtype=float)


def fractional_difference(values: Sequence[float], d: float, max_lags: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Apply causal fractional differencing and return transformed values and weights."""

    y = np.asarray(values, dtype=float).reshape(-1)
    weights = fractional_diff_weights(d, max_lags=max_lags)
    z = np.convolve(y, weights, mode="full")[: len(y)]
    return z, weights


def inverse_fractional_forecast(
    history: Sequence[float],
    differenced_forecast: Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    """Invert z_t = sum_k w_k y_{t-k} recursively for future y values."""

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


def estimate_gph_d(values: Sequence[float], m: Optional[int] = None) -> Dict[str, float]:
    """Geweke-Porter-Hudak log-periodogram estimator."""

    y = np.asarray(values, dtype=float).reshape(-1)
    y = y[np.isfinite(y)] - np.nanmean(y)
    n = len(y)
    if n < 64:
        raise ValueError("GPH estimation needs at least 64 observations")
    freqs, pxx = signal.periodogram(y)
    freqs = freqs[1:]
    pxx = np.maximum(pxx[1:], 1e-18)
    if m is None:
        m = max(20, int(n**0.5))
    m = min(int(m), len(freqs))
    lam = 2.0 * np.pi * np.arange(1, m + 1) / n
    x = np.log(4.0 * np.sin(lam / 2.0) ** 2)
    target = np.log(pxx[:m])
    slope, intercept, r_value, p_value, stderr = stats.linregress(x, target)
    d_hat = float(-0.5 * slope)
    return {
        "d": d_hat,
        "bandwidth": float(m),
        "intercept": float(intercept),
        "slope": float(slope),
        "stderr": float(stderr),
        "r2": float(r_value**2),
        "p_value": float(p_value),
    }


def estimate_local_whittle_d(
    values: Sequence[float],
    m: Optional[int] = None,
    bounds: Tuple[float, float] = (-0.45, 0.95),
) -> Dict[str, float]:
    """Local Whittle estimator for long-memory d."""

    y = np.asarray(values, dtype=float).reshape(-1)
    y = y[np.isfinite(y)] - np.nanmean(y)
    n = len(y)
    if n < 64:
        raise ValueError("Local Whittle estimation needs at least 64 observations")
    freqs, pxx = signal.periodogram(y)
    pxx = np.maximum(pxx[1:], 1e-18)
    if m is None:
        m = max(20, int(n**0.65))
    m = min(int(m), len(pxx))
    lam = 2.0 * np.pi * np.arange(1, m + 1) / n
    periodogram = pxx[:m]

    def objective(d_value: float) -> float:
        adjusted = periodogram * np.power(lam, 2.0 * d_value)
        return float(np.log(np.mean(adjusted)) - 2.0 * d_value * np.mean(np.log(lam)))

    opt = optimize.minimize(lambda z: objective(float(z[0])), x0=[0.25], bounds=[bounds], method="L-BFGS-B")
    return {
        "d": float(opt.x[0]),
        "bandwidth": float(m),
        "objective": float(opt.fun),
        "success": float(bool(opt.success)),
    }


@dataclass
class ARFIMAConfig:
    p: int = 1
    d: Optional[float] = None
    q: int = 1
    d_estimator: str = "local_whittle"  # local_whittle or gph
    max_fracdiff_lags: int = 4096
    trend: Optional[str] = "c"
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    fit_kwargs: Dict[str, Any] = field(default_factory=lambda: {"disp": False, "maxiter": 300})


class ARFIMAForecaster:
    """Approximate ARFIMA(p,d,q) forecaster with fractional inverse forecasting."""

    def __init__(self, config: ARFIMAConfig | None = None) -> None:
        self.config = config or ARFIMAConfig()
        self.series_: Optional[pd.Series] = None
        self.d_: Optional[float] = None
        self.d_metadata_: Dict[str, float] = {}
        self.weights_: Optional[np.ndarray] = None
        self.results_: Any = None
        self.residuals_: Optional[np.ndarray] = None

    def _estimate_d(self, values: np.ndarray) -> Tuple[float, Dict[str, float]]:
        if self.config.d is not None:
            return float(self.config.d), {"method": "fixed", "d": float(self.config.d)}
        if self.config.d_estimator == "gph":
            meta = estimate_gph_d(values)
        elif self.config.d_estimator == "local_whittle":
            meta = estimate_local_whittle_d(values)
        else:
            raise ValueError("d_estimator must be 'local_whittle' or 'gph'")
        meta["method"] = self.config.d_estimator
        return float(meta["d"]), meta

    def fit(
        self,
        data: pd.Series | pd.DataFrame | Sequence[float],
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
    ) -> "ARFIMAForecaster":
        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        d, d_meta = self._estimate_d(y.to_numpy(dtype=float))
        z, weights = fractional_difference(y.to_numpy(dtype=float), d=d, max_lags=self.config.max_fracdiff_lags)
        sm_sarimax = require_package("statsmodels.tsa.statespace.sarimax", "statsmodels")
        z_series = pd.Series(z, index=y.index, name="fracdiff_y").replace([np.inf, -np.inf], np.nan).dropna()
        model = sm_sarimax.SARIMAX(
            z_series,
            order=(int(self.config.p), 0, int(self.config.q)),
            trend=self.config.trend,
            enforce_stationarity=self.config.enforce_stationarity,
            enforce_invertibility=self.config.enforce_invertibility,
        )
        self.results_ = model.fit(**self.config.fit_kwargs)
        self.series_ = y
        self.d_ = d
        self.d_metadata_ = d_meta
        self.weights_ = weights
        self.residuals_ = np.asarray(self.results_.resid, dtype=float)
        return self

    def forecast(self, horizon: int, alpha: float = 0.05) -> ForecastResult:
        if self.results_ is None or self.series_ is None or self.weights_ is None or self.d_ is None:
            raise RuntimeError("fit the forecaster before forecast")
        z_pred = np.asarray(self.results_.get_forecast(steps=int(horizon)).predicted_mean, dtype=float)
        y_pred = inverse_fractional_forecast(self.series_.to_numpy(dtype=float), z_pred, self.weights_)
        lower, upper = empirical_interval(y_pred, self.residuals_, alpha=alpha, horizon_scale=True)
        return ForecastResult(
            mean=y_pred,
            index=future_index(self.series_.index, horizon),
            lower=lower,
            upper=upper,
            metadata={
                "model": "ARFIMA",
                "p": int(self.config.p),
                "d": float(self.d_),
                "q": int(self.config.q),
                "d_estimation": self.d_metadata_,
                "aic": float(getattr(self.results_, "aic", np.nan)),
                "bic": float(getattr(self.results_, "bic", np.nan)),
                "alpha": float(alpha),
            },
        )

    def save(self, path: str) -> None:
        ensure_dir(path)
        save_pickle(self, os.path.join(path, "model.pkl"))
        save_json(
            {
                "model": "ARFIMA",
                "config": self.config.__dict__,
                "d_metadata": self.d_metadata_,
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "ARFIMAForecaster":
        return load_pickle(os.path.join(path, "model.pkl"))
