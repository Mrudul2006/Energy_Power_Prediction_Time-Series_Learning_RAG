"""
MODEL 18 - Conformal prediction wrappers.

Conformal prediction turns point forecasts or quantile forecasts into calibrated
intervals. This file includes split conformal, normalized conformal, rolling
block conformal, and adaptive conformal updates for non-stationary residuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import os

import numpy as np
import pandas as pd

from .common import ForecastResult, ensure_dir, load_pickle, save_json, save_pickle


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Finite-sample split-conformal quantile."""

    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("scores cannot be empty")
    n = arr.size
    rank = int(np.ceil((n + 1) * (1.0 - float(alpha)))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(np.sort(arr)[rank])


@dataclass
class SplitConformalCalibrator:
    alpha: float = 0.10
    symmetric: bool = True
    scores_: Optional[np.ndarray] = None
    lower_score_: Optional[float] = None
    upper_score_: Optional[float] = None

    def fit(self, y_true: Sequence[float], y_pred: Sequence[float]) -> "SplitConformalCalibrator":
        y = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        if y.shape != pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        resid = y - pred
        if self.symmetric:
            self.scores_ = np.abs(resid)
            q = conformal_quantile(self.scores_, self.alpha)
            self.lower_score_ = q
            self.upper_score_ = q
        else:
            self.lower_score_ = conformal_quantile(pred - y, self.alpha / 2.0)
            self.upper_score_ = conformal_quantile(y - pred, self.alpha / 2.0)
            self.scores_ = resid
        return self

    def interval(self, y_pred: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        if self.lower_score_ is None or self.upper_score_ is None:
            raise RuntimeError("fit the calibrator before interval")
        pred = np.asarray(y_pred, dtype=float)
        return pred - float(self.lower_score_), pred + float(self.upper_score_)

    def apply(self, result: ForecastResult) -> ForecastResult:
        lower, upper = self.interval(result.mean)
        result.lower = lower
        result.upper = upper
        result.metadata["conformal"] = {
            "method": "split",
            "alpha": float(self.alpha),
            "symmetric": bool(self.symmetric),
        }
        return result


@dataclass
class NormalizedConformalCalibrator:
    """Conformal interval with scale model sigma_hat."""

    alpha: float = 0.10
    q_: Optional[float] = None

    def fit(self, y_true: Sequence[float], y_pred: Sequence[float], scale: Sequence[float]) -> "NormalizedConformalCalibrator":
        y = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        sigma = np.maximum(np.asarray(scale, dtype=float), 1e-8)
        scores = np.abs(y - pred) / sigma
        self.q_ = conformal_quantile(scores, self.alpha)
        return self

    def interval(self, y_pred: Sequence[float], scale: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        if self.q_ is None:
            raise RuntimeError("fit the calibrator before interval")
        pred = np.asarray(y_pred, dtype=float)
        sigma = np.maximum(np.asarray(scale, dtype=float), 1e-8)
        width = float(self.q_) * sigma
        return pred - width, pred + width


@dataclass
class BlockConformalCalibrator:
    """
    Rolling block conformal calibration for autocorrelated residuals.

    Scores are the max absolute error within each calibration block, so the
    resulting interval protects a horizon path rather than a single point.
    """

    alpha: float = 0.10
    block_size: int = 24
    q_: Optional[float] = None

    def fit(self, residuals: Sequence[float]) -> "BlockConformalCalibrator":
        resid = np.asarray(residuals, dtype=float)
        resid = resid[np.isfinite(resid)]
        if resid.size < self.block_size:
            raise ValueError("not enough residuals for the requested block_size")
        scores = []
        for start in range(0, resid.size - self.block_size + 1):
            scores.append(float(np.max(np.abs(resid[start : start + self.block_size]))))
        self.q_ = conformal_quantile(scores, self.alpha)
        return self

    def interval(self, y_pred: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        if self.q_ is None:
            raise RuntimeError("fit the calibrator before interval")
        pred = np.asarray(y_pred, dtype=float)
        return pred - float(self.q_), pred + float(self.q_)


@dataclass
class AdaptiveConformalCalibrator:
    """
    Online adaptive conformal quantile update.

    q_{t+1} = max(0, q_t + eta * (miss_t - alpha))
    where miss_t is 1 if y_t was outside the current interval.
    """

    alpha: float = 0.10
    learning_rate: float = 0.01
    q_: float = 0.0
    history_: list[Dict[str, float]] = field(default_factory=list)

    def initialize(self, residuals: Sequence[float]) -> "AdaptiveConformalCalibrator":
        self.q_ = conformal_quantile(np.abs(residuals), self.alpha)
        return self

    def interval(self, y_pred: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        pred = np.asarray(y_pred, dtype=float)
        return pred - self.q_, pred + self.q_

    def update(self, y_true: float, y_pred: float) -> "AdaptiveConformalCalibrator":
        miss = float(abs(float(y_true) - float(y_pred)) > self.q_)
        self.q_ = float(max(0.0, self.q_ + self.learning_rate * (miss - self.alpha)))
        self.history_.append({"y_true": float(y_true), "y_pred": float(y_pred), "miss": miss, "q": self.q_})
        return self


class ConformalForecaster:
    """
    Wrapper for any base forecaster exposing fit(...) and forecast(...).

    The calibration split should be chronologically after the training split.
    """

    def __init__(self, base_forecaster: Any, calibrator: Optional[SplitConformalCalibrator] = None) -> None:
        self.base_forecaster = base_forecaster
        self.calibrator = calibrator or SplitConformalCalibrator()
        self.calibrated_ = False

    def fit(self, train_data, *args, **kwargs) -> "ConformalForecaster":
        self.base_forecaster.fit(train_data, *args, **kwargs)
        return self

    def calibrate(self, y_true: Sequence[float], y_pred: Sequence[float]) -> "ConformalForecaster":
        self.calibrator.fit(y_true, y_pred)
        self.calibrated_ = True
        return self

    def forecast(self, horizon: int, *args, **kwargs) -> ForecastResult:
        if not self.calibrated_:
            raise RuntimeError("call calibrate before forecast")
        result = self.base_forecaster.forecast(horizon, *args, **kwargs)
        return self.calibrator.apply(result)

    def save(self, path: str) -> None:
        ensure_dir(path)
        save_pickle(self, os.path.join(path, "model.pkl"))
        save_json(
            {
                "model": "Conformal",
                "base_model": self.base_forecaster.__class__.__name__,
                "calibrated": bool(self.calibrated_),
            },
            os.path.join(path, "metadata.json"),
        )

    @classmethod
    def load(cls, path: str) -> "ConformalForecaster":
        return load_pickle(os.path.join(path, "model.pkl"))
