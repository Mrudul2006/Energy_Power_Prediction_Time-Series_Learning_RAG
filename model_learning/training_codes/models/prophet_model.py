"""
MODEL 3 - Facebook/Meta Prophet.

Prophet is an optional dependency. This wrapper keeps the DataFrame plumbing,
India holiday support, extra regressors, and ForecastResult conversion in one
place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence
import os

import numpy as np
import pandas as pd

from .common import ForecastResult, ensure_dir, ensure_series, future_index, load_pickle, require_package, save_json, save_pickle


@dataclass
class ProphetConfig:
    growth: str = "linear"
    yearly_seasonality: str | bool | int = "auto"
    weekly_seasonality: str | bool | int = "auto"
    daily_seasonality: str | bool | int = "auto"
    seasonality_mode: str = "additive"
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    holidays_prior_scale: float = 10.0
    interval_width: float = 0.95
    add_india_holidays: bool = True
    extra_seasonalities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regressors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)


class ProphetForecaster:
    """Prophet forecaster with future-known regressors and holiday support."""

    def __init__(self, config: ProphetConfig | None = None) -> None:
        self.config = config or ProphetConfig()
        self.model_: Any = None
        self.series_: Optional[pd.Series] = None
        self.regressor_columns_: list[str] = []

    def _make_training_frame(self, y: pd.Series, regressors: Optional[pd.DataFrame]) -> pd.DataFrame:
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("Prophet requires a DatetimeIndex or timestamp_col")
        frame = pd.DataFrame({"ds": y.index, "y": y.to_numpy(dtype=float)})
        if regressors is not None:
            reg = regressors.copy()
            if not reg.index.equals(y.index):
                reg = reg.reindex(y.index)
            for col in reg.columns:
                frame[col] = pd.to_numeric(reg[col], errors="coerce").to_numpy(dtype=float)
        return frame.dropna()

    def fit(
        self,
        data: pd.Series | pd.DataFrame | Sequence[float],
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        regressors: Optional[pd.DataFrame] = None,
    ) -> "ProphetForecaster":
        prophet_mod = require_package("prophet", "prophet")
        y = ensure_series(data, target_col=target_col, timestamp_col=timestamp_col)
        model = prophet_mod.Prophet(
            growth=self.config.growth,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            seasonality_mode=self.config.seasonality_mode,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            holidays_prior_scale=self.config.holidays_prior_scale,
            interval_width=self.config.interval_width,
        )
        if self.config.add_india_holidays:
            model.add_country_holidays(country_name="IN")
        for name, kwargs in self.config.extra_seasonalities.items():
            model.add_seasonality(name=name, **kwargs)
        reg_cols = list(regressors.columns) if regressors is not None else list(self.config.regressors.keys())
        for col in reg_cols:
            model.add_regressor(col, **self.config.regressors.get(col, {}))
        train = self._make_training_frame(y, regressors)
        model.fit(train, **self.config.fit_kwargs)
        self.model_ = model
        self.series_ = y
        self.regressor_columns_ = reg_cols
        return self

    def forecast(
        self,
        horizon: int,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> ForecastResult:
        if self.model_ is None or self.series_ is None:
            raise RuntimeError("fit the forecaster before forecast")
        index = future_index(self.series_.index, horizon)
        future = pd.DataFrame({"ds": index})
        if self.regressor_columns_:
            if future_regressors is None:
                raise ValueError("future_regressors are required because regressors were used during fit")
            reg = future_regressors.copy()
            if not reg.index.equals(index):
                reg = reg.reindex(index)
            missing = [c for c in self.regressor_columns_ if c not in reg.columns]
            if missing:
                raise ValueError(f"future_regressors missing columns: {missing}")
            for col in self.regressor_columns_:
                future[col] = pd.to_numeric(reg[col], errors="coerce").to_numpy(dtype=float)
        fcst = self.model_.predict(future)
        component_cols = [c for c in ("trend", "yearly", "weekly", "daily", "holidays") if c in fcst.columns]
        return ForecastResult(
            mean=fcst["yhat"].to_numpy(dtype=float),
            index=index,
            lower=fcst["yhat_lower"].to_numpy(dtype=float),
            upper=fcst["yhat_upper"].to_numpy(dtype=float),
            components={c: fcst[c].to_numpy(dtype=float) for c in component_cols},
            metadata={"model": "Prophet", "interval_width": float(self.config.interval_width)},
        )

    def save(self, path: str) -> None:
        ensure_dir(path)
        save_pickle(self, os.path.join(path, "model.pkl"))
        save_json(
            {
                "model": "Prophet",
                "config": self.config.__dict__,
                "regressors": list(self.regressor_columns_),
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "ProphetForecaster":
        return load_pickle(os.path.join(path, "model.pkl"))
