"""
MODEL 4 - Exponential Smoothing State Space Models (ETS).

Uses statsmodels' ETSModel when available. The wrapper exposes the standard
Error/Trend/Seasonal configuration and returns forecast intervals either from
statsmodels prediction results or residual empirical quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence
import os

import numpy as np
import pandas as pd

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


@dataclass
class ETSConfig:
    error: str = "add"
    trend: Optional[str] = "add"
    damped_trend: bool = False
    seasonal: Optional[str] = "add"
    seasonal_periods: int = 24
    initialization_method: str = "estimated"
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)


class ETSForecaster:
    """State-space ETS forecaster for a single seasonal period."""

    def __init__(self, config: ETSConfig | None = None) -> None:
        self.config = config or ETSConfig()
        self.series_: Optional[pd.Series] = None
        self.model_: Any = None
        self.results_: Any = None
        self.residuals_: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.Series | pd.DataFrame | Sequence[float],
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
    ) -> "ETSForecaster":
        ets_mod = require_package("statsmodels.tsa.exponential_smoothing.ets", "statsmodels")
        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        model = ets_mod.ETSModel(
            y,
            error=self.config.error,
            trend=self.config.trend,
            damped_trend=self.config.damped_trend,
            seasonal=self.config.seasonal,
            seasonal_periods=self.config.seasonal_periods,
            initialization_method=self.config.initialization_method,
        )
        self.results_ = model.fit(**self.config.fit_kwargs)
        self.model_ = model
        self.series_ = y
        fitted = np.asarray(self.results_.fittedvalues, dtype=float)
        self.residuals_ = y.to_numpy(dtype=float)[-len(fitted) :] - fitted
        return self

    def forecast(self, horizon: int, alpha: float = 0.05) -> ForecastResult:
        if self.results_ is None or self.series_ is None:
            raise RuntimeError("fit the forecaster before forecast")
        index = future_index(self.series_.index, horizon)
        mean = np.asarray(self.results_.forecast(steps=int(horizon)), dtype=float)
        lower = upper = None
        try:
            pred = self.results_.get_prediction(start=len(self.series_), end=len(self.series_) + horizon - 1)
            conf = pred.conf_int(alpha=alpha)
            lower = np.asarray(conf.iloc[:, 0], dtype=float)
            upper = np.asarray(conf.iloc[:, 1], dtype=float)
        except Exception:
            residuals = self.residuals_ if self.residuals_ is not None else []
            lower, upper = empirical_interval(mean, residuals, alpha=alpha)
        return ForecastResult(
            mean=mean,
            index=index,
            lower=lower,
            upper=upper,
            metadata={
                "model": "ETS",
                "error": self.config.error,
                "trend": self.config.trend,
                "seasonal": self.config.seasonal,
                "seasonal_periods": int(self.config.seasonal_periods),
                "aic": float(getattr(self.results_, "aic", np.nan)),
                "bic": float(getattr(self.results_, "bic", np.nan)),
            },
        )

    def save(self, path: str) -> None:
        ensure_dir(path)
        save_pickle(self, os.path.join(path, "model.pkl"))
        save_json(
            {
                "model": "ETS",
                "config": self.config.__dict__,
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "ETSForecaster":
        return load_pickle(os.path.join(path, "model.pkl"))
