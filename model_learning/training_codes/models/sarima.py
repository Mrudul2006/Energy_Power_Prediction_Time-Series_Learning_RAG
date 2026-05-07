"""
MODEL 1 - SARIMA.

This module wraps statsmodels' SARIMAX implementation with an energy-focused
interface: order search, residual diagnostics, forecast intervals, and a
single ForecastResult output object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

from .common import ForecastResult, ensure_dir, ensure_series, future_index, load_pickle, require_package, save_json, save_pickle


@dataclass
class SARIMAConfig:
    order: Tuple[int, int, int] = (1, 0, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24)
    trend: Optional[str] = "c"
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    concentrate_scale: bool = True
    fit_kwargs: Dict[str, Any] = field(default_factory=lambda: {"disp": False, "maxiter": 300})


class SARIMAForecaster:
    """Seasonal ARIMA/SARIMAX forecaster for one dominant seasonality."""

    def __init__(self, config: SARIMAConfig | None = None) -> None:
        self.config = config or SARIMAConfig()
        self.series_: Optional[pd.Series] = None
        self.model_: Any = None
        self.results_: Any = None
        self.residuals_: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.Series | pd.DataFrame | Sequence[float],
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        exog: Optional[pd.DataFrame | np.ndarray] = None,
    ) -> "SARIMAForecaster":
        sm_sarimax = require_package("statsmodels.tsa.statespace.sarimax", "statsmodels")
        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        model = sm_sarimax.SARIMAX(
            y,
            exog=exog,
            order=self.config.order,
            seasonal_order=self.config.seasonal_order,
            trend=self.config.trend,
            enforce_stationarity=self.config.enforce_stationarity,
            enforce_invertibility=self.config.enforce_invertibility,
            concentrate_scale=self.config.concentrate_scale,
        )
        self.results_ = model.fit(**self.config.fit_kwargs)
        self.model_ = model
        self.series_ = y
        self.residuals_ = np.asarray(self.results_.resid, dtype=float)
        return self

    def forecast(
        self,
        horizon: int,
        future_exog: Optional[pd.DataFrame | np.ndarray] = None,
        alpha: float = 0.05,
    ) -> ForecastResult:
        if self.results_ is None or self.series_ is None:
            raise RuntimeError("fit the forecaster before forecast")
        pred = self.results_.get_forecast(steps=int(horizon), exog=future_exog)
        mean = np.asarray(pred.predicted_mean, dtype=float)
        conf = pred.conf_int(alpha=alpha)
        index = future_index(self.series_.index, horizon)
        return ForecastResult(
            mean=mean,
            index=index,
            lower=np.asarray(conf.iloc[:, 0], dtype=float),
            upper=np.asarray(conf.iloc[:, 1], dtype=float),
            metadata={
                "model": "SARIMA",
                "order": self.config.order,
                "seasonal_order": self.config.seasonal_order,
                "aic": float(getattr(self.results_, "aic", np.nan)),
                "bic": float(getattr(self.results_, "bic", np.nan)),
                "alpha": float(alpha),
            },
        )

    def diagnostics(self, lags: Sequence[int] = (1, 24, 48, 168)) -> Dict[str, Any]:
        """Return compact residual diagnostics for stationarity and whiteness checks."""

        if self.residuals_ is None:
            raise RuntimeError("fit the forecaster before diagnostics")
        resid = self.residuals_[np.isfinite(self.residuals_)]
        sm_diag = require_package("statsmodels.stats.diagnostic", "statsmodels")
        lb = sm_diag.acorr_ljungbox(resid, lags=list(lags), return_df=True)
        return {
            "residual_mean": float(np.mean(resid)),
            "residual_std": float(np.std(resid)),
            "residual_skew": float(stats.skew(resid)),
            "residual_kurtosis": float(stats.kurtosis(resid, fisher=True)),
            "ljung_box": lb.to_dict(orient="index"),
            "aic": float(getattr(self.results_, "aic", np.nan)),
            "bic": float(getattr(self.results_, "bic", np.nan)),
        }

    def save(self, path: str) -> None:
        ensure_dir(path)
        save_pickle(self, os.path.join(path, "model.pkl"))
        save_json(
            {
                "model": "SARIMA",
                "config": self.config.__dict__,
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "SARIMAForecaster":
        return load_pickle(os.path.join(path, "model.pkl"))

    @staticmethod
    def search_orders(
        data: pd.Series | pd.DataFrame | Sequence[float],
        p: Iterable[int] = range(0, 3),
        d: Iterable[int] = range(0, 2),
        q: Iterable[int] = range(0, 3),
        P: Iterable[int] = range(0, 2),
        D: Iterable[int] = range(0, 2),
        Q: Iterable[int] = range(0, 2),
        seasonal_period: int = 24,
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        criterion: str = "aic",
        max_models: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Grid-search SARIMA orders and return sorted scores.

        This intentionally favors a transparent exhaustive search over
        auto-arima black boxes. It is suitable for a constrained candidate
        grid; large grids can be slow.
        """

        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        sm_sarimax = require_package("statsmodels.tsa.statespace.sarimax", "statsmodels")
        rows = []
        candidates = itertools.product(p, d, q, P, D, Q)
        for i, (pi, di, qi, Pi, Di, Qi) in enumerate(candidates):
            if max_models is not None and i >= max_models:
                break
            order = (int(pi), int(di), int(qi))
            seasonal_order = (int(Pi), int(Di), int(Qi), int(seasonal_period))
            try:
                model = sm_sarimax.SARIMAX(
                    y,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    concentrate_scale=True,
                )
                res = model.fit(disp=False, maxiter=200)
                rows.append(
                    {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "aic": float(res.aic),
                        "bic": float(res.bic),
                        "hqic": float(res.hqic),
                    }
                )
            except Exception as exc:  # statsmodels raises many optimizer-specific errors
                rows.append({"order": order, "seasonal_order": seasonal_order, "error": str(exc)})
        frame = pd.DataFrame(rows)
        if criterion in frame.columns:
            frame = frame.sort_values(criterion, na_position="last").reset_index(drop=True)
        return frame
