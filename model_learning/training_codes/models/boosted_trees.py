"""
MODEL 6 - XGBoost / LightGBM gradient-boosted trees.

This file implements an energy-focused tabular forecaster. It turns the time
series into lag, rolling, and future-known calendar features, then fits either:
- XGBoost, if installed
- LightGBM, if installed
- sklearn HistGradientBoostingRegressor fallback

Both direct multi-horizon and recursive forecasting modes are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from .common import (
    ForecastResult,
    empirical_interval,
    ensure_dir,
    ensure_series,
    future_index,
    lag_feature_row,
    load_pickle,
    require_package,
    save_json,
    save_pickle,
    supervised_lag_frame,
)


@dataclass
class BoostedTreesConfig:
    model_type: str = "lightgbm"  # lightgbm, xgboost, sklearn
    mode: str = "direct"  # direct or recursive
    lags: Sequence[int] = (1, 2, 3, 24, 25, 48, 72, 168)
    rolling_windows: Sequence[int] = (3, 6, 24, 168)
    horizon: int = 24
    calendar: bool = True
    random_state: int = 7
    model_params: Dict[str, Any] = field(default_factory=dict)


class BoostedTreesForecaster:
    """Gradient-boosted tree forecaster with explicit temporal features."""

    def __init__(self, config: BoostedTreesConfig | None = None) -> None:
        self.config = config or BoostedTreesConfig()
        self.series_: Optional[pd.Series] = None
        self.models_: Dict[int, Any] = {}
        self.feature_columns_: List[str] = []
        self.residuals_: Dict[int, np.ndarray] = {}
        self.backend_: str = ""

    def _make_estimator(self) -> Any:
        params = dict(self.config.model_params)
        model_type = self.config.model_type.lower()
        if model_type == "xgboost":
            try:
                xgb = require_package("xgboost", "xgboost")
                defaults = {
                    "n_estimators": 800,
                    "max_depth": 6,
                    "learning_rate": 0.03,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "objective": "reg:squarederror",
                    "random_state": self.config.random_state,
                    "n_jobs": -1,
                }
                defaults.update(params)
                self.backend_ = "xgboost"
                return xgb.XGBRegressor(**defaults)
            except ImportError:
                model_type = "sklearn"
        if model_type == "lightgbm":
            try:
                lgb = require_package("lightgbm", "lightgbm")
                defaults = {
                    "n_estimators": 1200,
                    "num_leaves": 64,
                    "learning_rate": 0.03,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "objective": "regression",
                    "random_state": self.config.random_state,
                    "n_jobs": -1,
                }
                defaults.update(params)
                self.backend_ = "lightgbm"
                return lgb.LGBMRegressor(**defaults)
            except ImportError:
                model_type = "sklearn"
        defaults = {
            "max_iter": 600,
            "learning_rate": 0.035,
            "max_leaf_nodes": 63,
            "l2_regularization": 0.01,
            "random_state": self.config.random_state,
        }
        defaults.update(params)
        self.backend_ = "sklearn_hist_gradient_boosting"
        return HistGradientBoostingRegressor(**defaults)

    def fit(
        self,
        data: pd.Series | pd.DataFrame | Sequence[float],
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
    ) -> "BoostedTreesForecaster":
        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        self.series_ = y
        base_estimator = self._make_estimator()
        horizons = [1] if self.config.mode == "recursive" else list(range(1, int(self.config.horizon) + 1))
        for h in horizons:
            X, target = supervised_lag_frame(
                y,
                lags=self.config.lags,
                rolling_windows=self.config.rolling_windows,
                horizon=h,
                calendar=self.config.calendar,
            )
            estimator = clone(base_estimator)
            estimator.fit(X, target)
            pred = np.asarray(estimator.predict(X), dtype=float)
            self.models_[h] = estimator
            self.residuals_[h] = target.to_numpy(dtype=float) - pred
            if not self.feature_columns_:
                self.feature_columns_ = X.columns.tolist()
        return self

    def _predict_row(self, model: Any, history: Sequence[float], timestamp: Any) -> float:
        row = lag_feature_row(history, timestamp, self.config.lags, self.config.rolling_windows)
        row = row.reindex(columns=self.feature_columns_, fill_value=0.0)
        return float(np.asarray(model.predict(row), dtype=float).reshape(-1)[0])

    def forecast(self, horizon: Optional[int] = None, alpha: float = 0.05) -> ForecastResult:
        if self.series_ is None or not self.models_:
            raise RuntimeError("fit the forecaster before forecast")
        h = int(horizon or self.config.horizon)
        index = future_index(self.series_.index, h)
        history = self.series_.to_numpy(dtype=float).tolist()
        preds: List[float] = []
        if self.config.mode == "recursive":
            model = self.models_[1]
            for ts in index:
                yhat = self._predict_row(model, history, ts)
                preds.append(yhat)
                history.append(yhat)
            residuals = self.residuals_[1]
        else:
            for step, ts in enumerate(index, start=1):
                model_key = min(step, max(self.models_))
                yhat = self._predict_row(self.models_[model_key], history, ts)
                preds.append(yhat)
            residuals = np.concatenate([self.residuals_.get(i, np.array([])) for i in range(1, min(h, max(self.models_)) + 1)])
        point = np.asarray(preds, dtype=float)
        lower, upper = empirical_interval(point, residuals, alpha=alpha)
        return ForecastResult(
            mean=point,
            index=index,
            lower=lower,
            upper=upper,
            metadata={
                "model": "BoostedTrees",
                "backend": self.backend_,
                "mode": self.config.mode,
                "lags": list(self.config.lags),
                "rolling_windows": list(self.config.rolling_windows),
                "alpha": float(alpha),
            },
        )

    def feature_importance(self, horizon: int = 1) -> pd.Series:
        """Return gain/split/importances when exposed by the backend estimator."""

        if horizon not in self.models_:
            raise ValueError(f"model for horizon {horizon} was not fitted")
        model = self.models_[horizon]
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        else:
            values = np.full(len(self.feature_columns_), np.nan)
        return pd.Series(values, index=self.feature_columns_).sort_values(ascending=False)

    def save(self, path: str) -> None:
        ensure_dir(path)
        save_pickle(self, os.path.join(path, "model.pkl"))
        save_json(
            {
                "model": "BoostedTrees",
                "config": self.config.__dict__,
                "backend": self.backend_,
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "BoostedTreesForecaster":
        return load_pickle(os.path.join(path, "model.pkl"))
