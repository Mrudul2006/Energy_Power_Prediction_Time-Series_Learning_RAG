"""
Online residual corrector for EnergyCast.
Uses Ridge regression on a sliding buffer of (features, error) pairs.
Persists state to disk so it survives server restarts.
"""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)

PERSIST_PATH = Path(__file__).parent / "online_learner.pkl"
BUFFER_MAX   = 168   # 1 week of hourly observations
MIN_SAMPLES  = 5     # min before Ridge activates
EMA_ALPHA    = 0.3   # smoothing for EMA fallback
RIDGE_ALPHA  = 10.0  # L2 regularisation


class OnlineLearner:
    def __init__(self) -> None:
        self.buffer_X: list[list[float]] = []
        self.buffer_y: list[float] = []
        self._ema: float = 0.0
        self._n: int = 0
        self._ridge = None

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, features: list[float], residual: float, save_to_disk: bool = True, refit: bool = True) -> None:
        """Record actual error and refit."""
        self.buffer_X.append(features)
        self.buffer_y.append(residual)
        if len(self.buffer_X) > BUFFER_MAX:
            self.buffer_X = self.buffer_X[-BUFFER_MAX:]
            self.buffer_y = self.buffer_y[-BUFFER_MAX:]
        self._ema = EMA_ALPHA * residual + (1 - EMA_ALPHA) * self._ema
        self._n += 1
        if refit:
            self._refit()
        if save_to_disk:
            self.save()

    def predict_correction(self, features: list[float]) -> float:
        """Return predicted residual correction for a feature vector."""
        if self._ridge is not None and len(self.buffer_X) >= MIN_SAMPLES:
            try:
                return float(self._ridge.predict([features])[0])
            except Exception:
                pass
        return self._ema

    @property
    def n_updates(self) -> int:
        return self._n

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        try:
            with open(PERSIST_PATH, "wb") as f:
                pickle.dump({
                    "buffer_X": self.buffer_X,
                    "buffer_y": self.buffer_y,
                    "_ema": self._ema,
                    "_n": self._n,
                }, f)
        except Exception as e:
            log.warning("OnlineLearner.save failed: %s", e)

    @classmethod
    def load(cls) -> "OnlineLearner":
        obj = cls()
        try:
            with open(PERSIST_PATH, "rb") as f:
                state = pickle.load(f)
            obj.buffer_X = state["buffer_X"]
            obj.buffer_y = state["buffer_y"]
            obj._ema     = state["_ema"]
            obj._n       = state["_n"]
            obj._refit()
            log.info("OnlineLearner loaded (%d samples)", len(obj.buffer_X))
        except FileNotFoundError:
            pass
        except Exception as e:
            log.warning("OnlineLearner.load failed: %s", e)
        return obj

    # ── internal ─────────────────────────────────────────────────────────────

    def _refit(self) -> None:
        if len(self.buffer_X) < MIN_SAMPLES:
            return
        try:
            from sklearn.linear_model import Ridge
            X = np.array(self.buffer_X)
            y = np.array(self.buffer_y)
            self._ridge = Ridge(alpha=RIDGE_ALPHA)
            self._ridge.fit(X, y)
        except Exception as e:
            log.warning("OnlineLearner._refit failed: %s", e)
            self._ridge = None
