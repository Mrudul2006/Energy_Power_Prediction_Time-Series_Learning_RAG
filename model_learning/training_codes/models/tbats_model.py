"""
MODEL 5 - TBATS.

If the optional `tbats` package is installed this wrapper delegates to its
TBATS estimator. Otherwise it uses a transparent TBATS-style fallback:
Box-Cox transform + Fourier terms for multiple seasonalities + ARMA errors via
statsmodels SARIMAX. The fallback is not a full trigonometric state-space
likelihood, but it preserves the modeling ingredients that matter for energy
forecasting and remains inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

from .common import ForecastResult, ensure_dir, ensure_series, future_index, load_pickle, require_package, save_json, save_pickle


def _boxcox(values: np.ndarray, lam: float, shift: float) -> np.ndarray:
    x = values + shift
    if np.any(x <= 0):
        raise ValueError("Box-Cox transformed values must be positive after shift")
    if abs(lam) < 1e-10:
        return np.log(x)
    return (np.power(x, lam) - 1.0) / lam


def _inv_boxcox(values: np.ndarray, lam: float, shift: float) -> np.ndarray:
    if abs(lam) < 1e-10:
        out = np.exp(values)
    else:
        out = np.power(np.maximum(lam * values + 1.0, 1e-12), 1.0 / lam)
    return out - shift


def tbats_fourier_matrix(
    n: int,
    seasonal_periods: Sequence[float],
    harmonics: Mapping[float, int],
    start: int = 0,
) -> pd.DataFrame:
    """Fourier design matrix used by the fallback model."""

    t = np.arange(start, start + int(n), dtype=float)
    cols: Dict[str, np.ndarray] = {}
    for period in seasonal_periods:
        period_f = float(period)
        h_count = int(harmonics.get(period, harmonics.get(period_f, 1)))
        for k in range(1, h_count + 1):
            angle = 2.0 * math.pi * k * t / period_f
            safe_period = str(period).replace(".", "_")
            cols[f"s{safe_period}_sin_{k}"] = np.sin(angle)
            cols[f"s{safe_period}_cos_{k}"] = np.cos(angle)
    return pd.DataFrame(cols)


@dataclass
class TBATSConfig:
    seasonal_periods: Sequence[float] = (24.0, 168.0, 8760.0)
    use_boxcox: bool = True
    boxcox_lambda: Optional[float] = None
    use_trend: bool = True
    use_damped_trend: bool = True
    use_arma_errors: bool = True
    arma_order: tuple[int, int] = (2, 2)
    harmonics: Optional[Dict[float, int]] = None
    max_harmonics: int = 10
    prefer_tbats_package: bool = True
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)


class TBATSForecaster:
    """Multiple-seasonality TBATS forecaster with package and fallback backends."""

    def __init__(self, config: TBATSConfig | None = None) -> None:
        self.config = config or TBATSConfig()
        self.series_: Optional[pd.Series] = None
        self.backend_: str = ""
        self.model_: Any = None
        self.results_: Any = None
        self.lambda_: float = 1.0
        self.shift_: float = 0.0
        self.harmonics_: Dict[float, int] = {}

    def _default_harmonics(self) -> Dict[float, int]:
        if self.config.harmonics is not None:
            return {float(k): int(v) for k, v in self.config.harmonics.items()}
        out: Dict[float, int] = {}
        for period in self.config.seasonal_periods:
            if period <= 24:
                out[float(period)] = min(self.config.max_harmonics, int(period // 2))
            elif period <= 168:
                out[float(period)] = min(self.config.max_harmonics, 8)
            else:
                out[float(period)] = min(self.config.max_harmonics, 5)
        return out

    def fit(
        self,
        data: pd.Series | pd.DataFrame | Sequence[float],
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
    ) -> "TBATSForecaster":
        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        self.series_ = y
        if self.config.prefer_tbats_package:
            try:
                tbats_mod = require_package("tbats", "tbats")
                estimator = tbats_mod.TBATS(
                    seasonal_periods=list(self.config.seasonal_periods),
                    use_box_cox=self.config.use_boxcox,
                    box_cox_bounds=(0, 1),
                    use_trend=self.config.use_trend,
                    use_damped_trend=self.config.use_damped_trend,
                    use_arma_errors=self.config.use_arma_errors,
                    **self.config.fit_kwargs,
                )
                self.results_ = estimator.fit(y.to_numpy(dtype=float))
                self.model_ = estimator
                self.backend_ = "tbats_package"
                return self
            except ImportError:
                pass

        self.backend_ = "fourier_sarimax_fallback"
        values = y.to_numpy(dtype=float)
        self.shift_ = float(max(0.0, 1.0 - np.nanmin(values)))
        if self.config.use_boxcox:
            self.lambda_ = (
                float(self.config.boxcox_lambda)
                if self.config.boxcox_lambda is not None
                else float(stats.boxcox_normmax(values + self.shift_, method="mle"))
            )
            transformed = _boxcox(values, self.lambda_, self.shift_)
        else:
            self.lambda_ = 1.0
            transformed = values
        self.harmonics_ = self._default_harmonics()
        exog = tbats_fourier_matrix(len(y), self.config.seasonal_periods, self.harmonics_, start=0)
        sm_sarimax = require_package("statsmodels.tsa.statespace.sarimax", "statsmodels")
        order = (int(self.config.arma_order[0]), 0, int(self.config.arma_order[1])) if self.config.use_arma_errors else (0, 0, 0)
        trend = "ct" if self.config.use_trend else "c"
        model = sm_sarimax.SARIMAX(
            transformed,
            exog=exog,
            order=order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.results_ = model.fit(disp=False, maxiter=300)
        self.model_ = model
        return self

    def forecast(self, horizon: int, alpha: float = 0.05) -> ForecastResult:
        if self.results_ is None or self.series_ is None:
            raise RuntimeError("fit the forecaster before forecast")
        index = future_index(self.series_.index, horizon)
        if self.backend_ == "tbats_package":
            point = np.asarray(self.results_.forecast(steps=int(horizon)), dtype=float)
            return ForecastResult(
                mean=point,
                index=index,
                metadata={"model": "TBATS", "backend": self.backend_, "seasonal_periods": list(self.config.seasonal_periods)},
            )

        exog_future = tbats_fourier_matrix(
            int(horizon),
            self.config.seasonal_periods,
            self.harmonics_,
            start=len(self.series_),
        )
        pred = self.results_.get_forecast(steps=int(horizon), exog=exog_future)
        mean_t = np.asarray(pred.predicted_mean, dtype=float)
        conf = np.asarray(pred.conf_int(alpha=alpha), dtype=float)
        if self.config.use_boxcox:
            mean = _inv_boxcox(mean_t, self.lambda_, self.shift_)
            lower = _inv_boxcox(conf[:, 0], self.lambda_, self.shift_)
            upper = _inv_boxcox(conf[:, 1], self.lambda_, self.shift_)
        else:
            mean, lower, upper = mean_t, conf[:, 0], conf[:, 1]
        return ForecastResult(
            mean=mean,
            index=index,
            lower=np.minimum(lower, upper),
            upper=np.maximum(lower, upper),
            metadata={
                "model": "TBATS",
                "backend": self.backend_,
                "seasonal_periods": list(self.config.seasonal_periods),
                "harmonics": self.harmonics_,
                "boxcox_lambda": float(self.lambda_),
                "aic": float(getattr(self.results_, "aic", np.nan)),
            },
        )

    def save(self, path: str) -> None:
        ensure_dir(path)
        save_pickle(self, os.path.join(path, "model.pkl"))
        save_json(
            {
                "model": "TBATS",
                "config": self.config.__dict__,
                "backend": self.backend_,
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "TBATSForecaster":
        return load_pickle(os.path.join(path, "model.pkl"))
