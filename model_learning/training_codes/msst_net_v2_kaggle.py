"""
MSST-Net v2 — cleaned_energy_data3.csv edition (Kaggle 2×T4 DDP).

What changed vs msst_net_kaggle.py
───────────────────────────────────
Dataset  : cleaned_energy_data3.csv  (52,608 rows · 71 columns · 6 years)
Target   : `consumption`  (MWh, hourly)

Data preprocessing added
  1. NaN fill  — lag/rolling warmup NaNs at series start are forward-then-back-filled.
  2. Extra feature normalization  — rolling_mean_24h, rolling_std_24h, diff_24h are
     z-score standardized using training-split statistics.

Architecture update
  hourly_features: 1 → 4
    channel 0 : BoxCox + fracdiff transformed consumption   (existing)
    channel 1 : rolling_mean_24h  (z-scored)  — recent level
    channel 2 : rolling_std_24h   (z-scored)  — recent volatility
    channel 3 : diff_24h          (z-scored)  — 24-h trend direction

  Everything else (daily encoder, weekly encoder, cross-resolution fusion,
  decoder, quantile head, loss, DDP, AMP) is identical to v1.

Performance metrics — printed every epoch and explained below
  MAE      Mean Absolute Error in MWh — same unit as target, easy to interpret.
  RMSE     Root Mean Squared Error    — penalises large spikes harder than MAE.
  MAPE     Mean Absolute % Error      — scale-free; e.g. 3.2 % means average
                                        forecast is 3.2 % off actual.
  Huber    Smooth MAE/MSE blend used as the training loss term.
  Pinball  Proper scoring rule for quantile forecasts (lower = better).
  Cov80    % of actuals inside the [Q10, Q90] predicted interval. Target ≈ 80 %.
  Winkler  Interval sharpness: width + penalty when actual falls outside.
           Lower = narrower and more accurate intervals.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scipy import signal, stats
from scipy.optimize import minimize
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# Statistical input transformation  (unchanged)
# ---------------------------------------------------------------------------

def _as_1d_float(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("expected at least one finite value")
    return arr


def estimate_boxcox_lambda(values: Sequence[float]) -> Tuple[float, float]:
    y = _as_1d_float(values)
    shift = 0.0
    if np.min(y) <= 0:
        shift = float(1.0 - np.min(y))
        y = y + shift
    lam = float(stats.boxcox_normmax(y, method="mle"))
    return lam, shift


def apply_boxcox(values: Sequence[float], lam: float, shift: float = 0.0) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    shifted = y + shift
    if np.any(shifted <= 0):
        raise ValueError("Box-Cox requires values + shift to be strictly positive")
    if abs(lam) < 1e-10:
        return np.log(shifted)
    return (np.power(shifted, lam) - 1.0) / lam


def invert_boxcox(values: Sequence[float], lam: float, shift: float = 0.0) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    if abs(lam) < 1e-10:
        y = np.exp(z)
    else:
        base = np.maximum(lam * z + 1.0, 1e-12)
        y = np.power(base, 1.0 / lam)
    return y - shift


def estimate_local_whittle_d(
    values: Sequence[float],
    bandwidth: Optional[int] = None,
    bounds: Tuple[float, float] = (-0.49, 0.99),
) -> Tuple[float, Dict[str, float]]:
    y = _as_1d_float(values)
    y = y - np.mean(y)
    n = len(y)
    if n < 64:
        raise ValueError("Local Whittle estimation needs at least 64 observations")
    _, power = signal.periodogram(y)
    power = np.maximum(power[1:], 1e-18)
    if bandwidth is None:
        bandwidth = min(len(power), max(20, int(n**0.65)))
    bandwidth = int(min(max(8, bandwidth), len(power)))
    lam = 2.0 * np.pi * np.arange(1, bandwidth + 1) / n
    periodogram = power[:bandwidth]

    def objective(d_value: float) -> float:
        adjusted = periodogram * np.power(lam, 2.0 * d_value)
        return float(np.log(np.mean(adjusted)) - 2.0 * d_value * np.mean(np.log(lam)))

    opt = minimize(lambda z: objective(float(z[0])), x0=[0.30], bounds=[bounds], method="L-BFGS-B")
    d_hat = float(opt.x[0])
    return d_hat, {"d": d_hat, "bandwidth": float(bandwidth),
                   "objective": float(opt.fun), "success": float(bool(opt.success))}


def fractional_diff_weights(d: float, threshold: float = 1e-5, max_size: int = 4096) -> np.ndarray:
    weights = [1.0]
    for k in range(1, max_size):
        w = -weights[-1] * (d - k + 1.0) / k
        weights.append(float(w))
        if abs(w) < threshold and k > 32:
            break
    return np.asarray(weights, dtype=float)


def fractional_difference_fixed_width(
    values: Sequence[float], d: float,
    threshold: float = 1e-5, max_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float).reshape(-1)
    weights = fractional_diff_weights(d, threshold=threshold, max_size=max_size)
    width = len(weights)
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) >= width:
        causal = np.convolve(x, weights, mode="full")[: len(x)]
        out[width - 1 :] = causal[width - 1 :]
    return out, weights


@dataclass
class BoxCoxFractionalDifferencer:
    fracdiff_threshold: float = 1e-5
    max_fracdiff_weights: int = 4096
    whittle_bandwidth: Optional[int] = None
    lam: Optional[float] = None
    shift: float = 0.0
    d: Optional[float] = None
    fracdiff_weights_: Optional[np.ndarray] = None
    metadata_: Dict[str, float] = field(default_factory=dict)

    def fit(self, values: Sequence[float]) -> "BoxCoxFractionalDifferencer":
        lam, shift = estimate_boxcox_lambda(values)
        z = apply_boxcox(values, lam, shift)
        d, whittle = estimate_local_whittle_d(z, bandwidth=self.whittle_bandwidth)
        transformed, weights = fractional_difference_fixed_width(
            z, d=d, threshold=self.fracdiff_threshold, max_size=self.max_fracdiff_weights)
        self.lam, self.shift, self.d, self.fracdiff_weights_ = lam, shift, d, weights
        self.metadata_ = {
            "boxcox_lambda": float(lam), "boxcox_shift": float(shift),
            "local_whittle_d": float(d), "fracdiff_width": float(len(weights)),
            "first_valid_index": float(np.argmax(np.isfinite(transformed))), **whittle,
        }
        return self

    @property
    def fitted(self) -> bool:
        return self.lam is not None and self.d is not None and self.fracdiff_weights_ is not None

    def transform(self, values: Sequence[float]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit the transformer before calling transform")
        z = apply_boxcox(values, float(self.lam), self.shift)
        transformed, _ = fractional_difference_fixed_width(
            z, d=float(self.d), threshold=self.fracdiff_threshold, max_size=self.max_fracdiff_weights)
        return transformed

    def fit_transform(self, values: Sequence[float]) -> np.ndarray:
        return self.fit(values).transform(values)


# ---------------------------------------------------------------------------
# Calendar and holiday covariates  (unchanged)
# ---------------------------------------------------------------------------

INDIAN_HOLIDAY_TYPES: Tuple[str, ...] = (
    "none", "republic_day", "holi", "good_friday", "eid_al_fitr", "ram_navami",
    "mahavir_jayanti", "buddha_purnima", "independence_day", "raksha_bandhan",
    "janmashtami", "gandhi_jayanti", "dussehra", "diwali", "christmas", "other",
)

DEFAULT_FIXED_INDIAN_HOLIDAYS: Mapping[Tuple[int, int], str] = {
    (1, 26): "republic_day", (8, 15): "independence_day",
    (10, 2): "gandhi_jayanti", (12, 25): "christmas",
}


class IndianHolidayCalendar:
    def __init__(self, extra_holidays: Optional[Mapping[object, str]] = None,
                 type_names: Sequence[str] = INDIAN_HOLIDAY_TYPES) -> None:
        self.type_names = tuple(type_names)
        self.type_to_id = {n: i for i, n in enumerate(self.type_names)}
        self.none_id = self.type_to_id.get("none", 0)
        self.other_id = self.type_to_id.get("other", len(self.type_names) - 1)
        self.extra: Dict[pd.Timestamp, str] = {}
        for k, v in (extra_holidays or {}).items():
            self.extra[pd.Timestamp(k).normalize()] = str(v)

    @property
    def num_types(self) -> int:
        return len(self.type_names)

    def _type_for_date(self, ts: pd.Timestamp) -> str:
        day = pd.Timestamp(ts).normalize()
        return self.extra.get(day, DEFAULT_FIXED_INDIAN_HOLIDAYS.get((day.month, day.day), "none"))

    def type_id_for_date(self, ts: pd.Timestamp) -> int:
        return self.type_to_id.get(self._type_for_date(ts), self.other_id)

    def features(self, timestamps: Sequence[pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray]:
        idx = pd.DatetimeIndex(timestamps)
        type_ids = np.asarray([self.type_id_for_date(ts) for ts in idx], dtype=np.int64)
        holiday = (type_ids != self.none_id).astype(np.float32)
        eve_ids = np.asarray([self.type_id_for_date(ts + pd.Timedelta(days=1)) for ts in idx], dtype=np.int64)
        eve = (eve_ids != self.none_id).astype(np.float32)
        return np.column_stack([holiday, eve]).astype(np.float32), type_ids


def fourier_calendar_features(timestamps: Sequence[pd.Timestamp]) -> np.ndarray:
    idx = pd.DatetimeIndex(timestamps)
    hod = idx.hour.to_numpy(dtype=float) + idx.minute.to_numpy(dtype=float) / 60.0
    how = idx.dayofweek.to_numpy(dtype=float) * 24.0 + hod
    hoy = (idx.dayofyear.to_numpy(dtype=float) - 1.0) * 24.0 + hod
    cols: List[np.ndarray] = []
    for k in (1, 2, 3):
        cols += [np.sin(2 * np.pi * k * hod / 24), np.cos(2 * np.pi * k * hod / 24)]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * how / 168), np.cos(2 * np.pi * k * how / 168)]
    cols += [np.sin(2 * np.pi * hoy / 8760), np.cos(2 * np.pi * hoy / 8760)]
    return np.column_stack(cols).astype(np.float32)


# ---------------------------------------------------------------------------
# Data preprocessing  (NEW)
# ---------------------------------------------------------------------------

_EXTRA_COLS = ["rolling_mean_24h", "rolling_std_24h", "diff_24h"]


def preprocess_rich_df(df: pd.DataFrame, timestamp_col: str, target_col: str = "") -> pd.DataFrame:
    """Sort, deduplicate, and ensure the three extra feature columns exist.

    If the source file already contains rolling_mean_24h / rolling_std_24h /
    diff_24h (e.g. cleaned_energy_data3.csv) those are NaN-filled as before.
    If they are absent (e.g. the Excel file which only has the raw consumption
    column) they are computed on-the-fly from *target_col*.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col])
    df = df.reset_index(drop=True)

    missing = [c for c in _EXTRA_COLS if c not in df.columns]
    if missing:
        if not target_col or target_col not in df.columns:
            raise ValueError(
                f"Columns {missing} are absent and target_col='{target_col}' "
                "is needed to compute them on-the-fly."
            )
        y = df[target_col].astype(float)
        df["rolling_mean_24h"] = y.rolling(24, min_periods=1).mean()
        df["rolling_std_24h"]  = y.rolling(24, min_periods=2).std().fillna(0.0)
        df["diff_24h"]         = y.diff(24).fillna(0.0)

    # Fill any residual NaNs (warmup rows in pre-computed files)
    df[_EXTRA_COLS] = df[_EXTRA_COLS].ffill().bfill()
    return df


def compute_feature_norm_stats(raw_y: np.ndarray, extra: np.ndarray) -> Dict[str, float]:
    """
    Compute z-score parameters for the 3 extra features using the full array.
    In _main_worker these are computed on the train split only, then applied to both.

    Returns dict with mean_i and std_i for i in 0..2.
    """
    stats_out: Dict[str, float] = {}
    for i in range(extra.shape[1]):
        col = extra[:, i]
        finite = col[np.isfinite(col)]
        stats_out[f"mean_{i}"] = float(np.mean(finite))
        stats_out[f"std_{i}"]  = float(np.std(finite) + 1e-8)
    return stats_out


def apply_feature_norm(extra: np.ndarray, norm_stats: Dict[str, float]) -> np.ndarray:
    out = extra.copy()
    for i in range(out.shape[1]):
        out[:, i] = (out[:, i] - norm_stats[f"mean_{i}"]) / norm_stats[f"std_{i}"]
    return out


# ---------------------------------------------------------------------------
# Dataset  (UPDATED — RichMultiScaleEnergyDataset)
# ---------------------------------------------------------------------------

def _hourly_frame(df: pd.DataFrame, timestamp_col: str, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out = out.sort_values(timestamp_col).set_index(timestamp_col)
    out = out[[target_col]].astype(float).resample("h").mean()
    out[target_col] = out[target_col].interpolate(limit_direction="both")
    if out[target_col].isna().any():
        raise ValueError("target column still contains NaNs after hourly interpolation")
    return out


def _block_stats(values: np.ndarray, block_size: int, count: int) -> np.ndarray:
    arr = values[-block_size * count :].reshape(count, block_size)
    return np.column_stack([
        np.mean(arr, axis=1), np.max(arr, axis=1),
        np.min(arr, axis=1),  np.std(arr, axis=1),
    ]).astype(np.float32)


class RichMultiScaleEnergyDataset(Dataset):
    """
    MSST-Net dataset for cleaned_energy_data3.csv.

    Hourly encoder input is 4-channel:
      [transformed_consumption, rolling_mean_24h_z, rolling_std_24h_z, diff_24h_z]

    The three extra channels are z-score normalised using statistics passed in
    via `norm_stats`. When `norm_stats=None` the stats are computed from this
    dataset itself (use only for the training split; pass the result to the
    validation split via `norm_stats=train_dataset.norm_stats`).
    """

    N_HOURLY_FEATURES = 4  # consumed by MSSTNetConfig.hourly_features

    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        target_col: str,
        horizon: int,
        transformer: Optional[BoxCoxFractionalDifferencer] = None,
        holiday_calendar: Optional[IndianHolidayCalendar] = None,
        stride: int = 1,
        fit_transformer: bool = True,
        norm_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)

        # ── Target ──────────────────────────────────────────────────────────
        self.frame = _hourly_frame(df, timestamp_col, target_col)
        self.target_col = target_col
        self.index = self.frame.index
        self.raw_y = self.frame[target_col].to_numpy(dtype=np.float32)

        self.transformer = transformer or BoxCoxFractionalDifferencer()
        if fit_transformer and not self.transformer.fitted:
            self.transformed_y = self.transformer.fit_transform(self.raw_y)
        else:
            self.transformed_y = self.transformer.transform(self.raw_y)
        self.transformed_y = self.transformed_y.astype(np.float32)

        # ── Extra features — aligned to hourly index ─────────────────────
        df_idx = df.copy()
        df_idx[timestamp_col] = pd.to_datetime(df_idx[timestamp_col])
        df_idx = df_idx.set_index(timestamp_col)[_EXTRA_COLS]
        df_idx = df_idx[~df_idx.index.duplicated(keep="first")]
        df_aligned = df_idx.reindex(self.frame.index).ffill().bfill()
        raw_extra = df_aligned.to_numpy(dtype=np.float32)  # (T, 3)

        # ── Normalisation ────────────────────────────────────────────────
        if norm_stats is None:
            norm_stats = compute_feature_norm_stats(self.raw_y, raw_extra)
        self.norm_stats = norm_stats
        self.extra_norm = apply_feature_norm(raw_extra, norm_stats).astype(np.float32)

        # ── Calendar ─────────────────────────────────────────────────────
        self.calendar = holiday_calendar or IndianHolidayCalendar()
        self.stride = max(1, int(stride))

        # ── Valid origins ────────────────────────────────────────────────
        finite = np.flatnonzero(np.isfinite(self.transformed_y))
        if len(finite) == 0:
            raise ValueError("transformed series has no finite values")
        self.first_valid = int(finite[0])
        self.min_history = max(168, 365 * 24, 52 * 168)
        first_origin = max(self.first_valid + self.min_history, self.min_history)
        last_origin  = len(self.raw_y) - self.horizon
        self.origins = list(range(first_origin, last_origin + 1, self.stride))
        if not self.origins:
            raise ValueError(
                f"Not enough history. Need ≥{self.min_history} hours + horizon {self.horizon}. "
                f"Got {len(self.raw_y)} rows.")

    def __len__(self) -> int:
        return len(self.origins)

    def _calendar_tensors(self, timestamps: Sequence[pd.Timestamp]) -> Dict[str, Tensor]:
        fourier = fourier_calendar_features(timestamps)
        flags, type_ids = self.calendar.features(timestamps)
        return {
            "fourier":       torch.from_numpy(fourier),
            "holiday_flags": torch.from_numpy(flags),
            "holiday_type":  torch.from_numpy(type_ids.astype(np.int64)),
        }

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        origin    = self.origins[item]
        origin_ts = self.index[origin]
        x         = self.transformed_y

        # ── Hourly window: shape (168, 4) ────────────────────────────────
        trans_hourly  = x[origin - 168 : origin, None].astype(np.float32)       # (168,1)
        extra_hourly  = self.extra_norm[origin - 168 : origin]                   # (168,3)
        hourly_values = np.concatenate([trans_hourly, extra_hourly], axis=1)    # (168,4)

        # ── Daily / weekly windows ────────────────────────────────────────
        daily_window  = x[origin - 365 * 24 : origin]
        weekly_window = x[origin - 52  * 168 : origin]
        if not (np.isfinite(hourly_values).all() and
                np.isfinite(daily_window).all() and
                np.isfinite(weekly_window).all()):
            raise RuntimeError("Non-finite values in input window at origin %d" % origin)

        daily_values = _block_stats(daily_window,  block_size=24,  count=365)
        weekly_stats = _block_stats(weekly_window, block_size=168, count=52)

        weekly_end_times = [
            origin_ts - pd.Timedelta(hours=168 * (52 - k)) + pd.Timedelta(hours=167)
            for k in range(52)
        ]
        weekly_doy = np.asarray(
            [pd.Timestamp(ts).dayofyear / 366.0 for ts in weekly_end_times], dtype=np.float32
        )[:, None]
        weekly_values = np.concatenate([weekly_stats, weekly_doy], axis=1).astype(np.float32)

        hourly_times  = self.index[origin - 168 : origin]
        daily_times   = [
            origin_ts - pd.Timedelta(hours=24 * (365 - k)) + pd.Timedelta(hours=23)
            for k in range(365)
        ]
        future_times  = self.index[origin : origin + self.horizon]

        out: Dict[str, Tensor] = {
            "hourly_values":  torch.from_numpy(hourly_values),
            "daily_values":   torch.from_numpy(daily_values),
            "weekly_values":  torch.from_numpy(weekly_values),
            "target":         torch.from_numpy(self.raw_y[origin : origin + self.horizon].astype(np.float32)),
            "origin_index":   torch.tensor(origin, dtype=torch.long),
        }
        for prefix, timestamps in (
            ("hourly",  hourly_times),
            ("daily",   daily_times),
            ("weekly",  weekly_end_times),
            ("future",  future_times),
        ):
            cal = self._calendar_tensors(timestamps)
            out[f"{prefix}_fourier"]       = cal["fourier"]
            out[f"{prefix}_holiday_flags"] = cal["holiday_flags"]
            out[f"{prefix}_holiday_type"]  = cal["holiday_type"]
        return out


# ---------------------------------------------------------------------------
# Neural architecture  (unchanged — hourly_features=4 set in config)
# ---------------------------------------------------------------------------

@dataclass
class MSSTNetConfig:
    horizon: int
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.10
    ff_multiplier: int = 4
    holiday_embedding_dim: int = 3
    holiday_type_count: int = len(INDIAN_HOLIDAY_TYPES)
    quantiles: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    hourly_length: int = 168
    daily_length: int = 365
    weekly_length: int = 52
    hourly_patch: int = 24
    daily_patch: int = 7
    weekly_patch: int = 4
    daily_patch_count: int = 52
    hourly_features: int = 4   # ← updated from 1 to 4
    daily_features: int = 4
    weekly_features: int = 5
    trend_windows: Tuple[int, int, int] = (24, 7, 4)
    acf_lag1: float = 0.70
    acf_lag24: float = 0.55
    acf_lag168: float = 0.35
    tau_hours: float = 6.0

    @property
    def fixed_fourier_dim(self) -> int:
        return 12

    @property
    def calendar_dim(self) -> int:
        return self.fixed_fourier_dim + 2 + self.holiday_embedding_dim


class CalendarEmbedding(nn.Module):
    def __init__(self, config: MSSTNetConfig) -> None:
        super().__init__()
        self.holiday_embedding = nn.Embedding(config.holiday_type_count, config.holiday_embedding_dim)

    def forward(self, fourier: Tensor, flags: Tensor, holiday_type: Tensor) -> Tensor:
        return torch.cat([fourier.float(), flags.float(),
                          self.holiday_embedding(holiday_type.long())], dim=-1)


class MovingAverageDecomposer(nn.Module):
    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = max(1, int(window))

    def forward(self, values: Tensor) -> Tuple[Tensor, Tensor]:
        if self.window <= 1:
            return values, values.new_zeros(values.shape)
        x = values.transpose(1, 2)
        left, right = self.window // 2, self.window - 1 - self.window // 2
        padded = F.pad(x, (left, right), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.window, stride=1).transpose(1, 2)
        return trend, values - trend


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim: int, patch_size: int, patch_count: int,
                 d_model: int, dropout: float) -> None:
        super().__init__()
        self.patch_size  = int(patch_size)
        self.patch_count = int(patch_count)
        self.proj    = nn.Linear(input_dim * self.patch_size, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        needed = self.patch_size * self.patch_count
        if x.size(1) < needed:
            raise ValueError(f"seq len {x.size(1)} < needed {needed}")
        x = x[:, -needed:, :]
        bsz, _, dim = x.shape
        x = x.reshape(bsz, self.patch_count, self.patch_size * dim)
        return self.dropout(self.norm(self.proj(x)))


class PeriodicAttentionBias(nn.Module):
    def __init__(self, token_count: int, patch_span_hours: float,
                 acf_lag1: float, acf_lag24: float, acf_lag168: float,
                 tau_hours: float) -> None:
        super().__init__()
        self.log_tau  = nn.Parameter(torch.tensor(math.log(max(tau_hours, 1e-3))))
        self.w_local  = nn.Parameter(torch.tensor(float(acf_lag1)))
        self.w_daily  = nn.Parameter(torch.tensor(float(acf_lag24)))
        self.w_weekly = nn.Parameter(torch.tensor(float(acf_lag168)))
        pos   = torch.arange(token_count, dtype=torch.float32) * patch_span_hours
        delta = torch.abs(pos[:, None] - pos[None, :])
        self.register_buffer("delta_hours", delta, persistent=False)
        self.register_buffer("daily_mask",  ((delta > 0) & (torch.remainder(delta,  24.0) < 1e-4)).float(), persistent=False)
        self.register_buffer("weekly_mask", ((delta > 0) & (torch.remainder(delta, 168.0) < 1e-4)).float(), persistent=False)

    def forward(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        delta = self.delta_hours.to(device=device, dtype=dtype)
        tau   = torch.exp(self.log_tau).to(device=device, dtype=dtype).clamp_min(1e-3)
        return (self.w_local.to(device, dtype)  * torch.exp(-delta / tau) +
                self.w_daily.to(device, dtype)  * self.daily_mask.to(device, dtype) +
                self.w_weekly.to(device, dtype) * self.weekly_mask.to(device, dtype))


class BiasedTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_multiplier * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_multiplier * d_model, d_model),
        )

    def forward(self, x: Tensor, attn_bias: Optional[Tensor] = None) -> Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_bias, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        return self.norm2(x + self.dropout(self.ff(x)))


class BiasedTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, ff_multiplier: int,
                 dropout: float, bias: Optional[PeriodicAttentionBias] = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [BiasedTransformerLayer(d_model, n_heads, ff_multiplier, dropout) for _ in range(n_layers)])
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias(x.device, x.dtype) if self.bias is not None else None
        for layer in self.layers:
            x = layer(x, bias)
        return x


class DualPathScaleEncoder(nn.Module):
    def __init__(self, config: MSSTNetConfig, value_dim: int, raw_length: int,
                 patch_size: int, patch_count: int, moving_average_window: int,
                 patch_span_hours: float) -> None:
        super().__init__()
        self.decomposer    = MovingAverageDecomposer(moving_average_window)
        token_dim          = value_dim + config.calendar_dim
        self.trend_patch   = PatchEmbedding(token_dim, patch_size, patch_count, config.d_model, config.dropout)
        self.seasonal_patch= PatchEmbedding(token_dim, patch_size, patch_count, config.d_model, config.dropout)
        self.trend_encoder = BiasedTransformerEncoder(
            config.d_model, config.n_heads, config.n_layers, config.ff_multiplier, config.dropout)
        self.seasonal_encoder = BiasedTransformerEncoder(
            config.d_model, config.n_heads, config.n_layers, config.ff_multiplier, config.dropout,
            bias=PeriodicAttentionBias(patch_count, patch_span_hours,
                                       config.acf_lag1, config.acf_lag24, config.acf_lag168, config.tau_hours))
        self.gate     = nn.Linear(config.d_model, config.d_model)
        self.out_norm = nn.LayerNorm(config.d_model)

    def forward(self, values: Tensor, calendar: Tensor) -> Tensor:
        trend, seasonal = self.decomposer(values)
        h_t = self.trend_encoder(self.trend_patch(torch.cat([trend,    calendar], dim=-1)))
        h_s = self.seasonal_encoder(self.seasonal_patch(torch.cat([seasonal, calendar], dim=-1)))
        g   = torch.sigmoid(self.gate(h_t))
        return self.out_norm(g * h_t + (1 - g) * h_s)


class FutureCovariateDecoder(nn.Module):
    def __init__(self, config: MSSTNetConfig) -> None:
        super().__init__()
        self.query_mlp = nn.Sequential(
            nn.Linear(config.calendar_dim, config.d_model), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.d_model, config.d_model),
        )
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.n_heads,
                                                 dropout=config.dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ff_multiplier * config.d_model), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.ff_multiplier * config.d_model, config.d_model),
        )
        self.norm2   = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, future_calendar: Tensor, encoder_context: Tensor) -> Tensor:
        q = self.query_mlp(future_calendar)
        a, _ = self.cross_attn(q, encoder_context, encoder_context, need_weights=False)
        x = self.norm1(q + self.dropout(a))
        return self.norm2(x + self.dropout(self.ff(x)))


class MonotoneQuantileHead(nn.Module):
    def __init__(self, d_model: int, n_quantiles: int) -> None:
        super().__init__()
        self.raw = nn.Linear(d_model, n_quantiles)

    def forward(self, x: Tensor) -> Tensor:
        raw   = self.raw(x)
        first = raw[..., :1]
        if raw.size(-1) == 1:
            return first
        return torch.cat([first, first + torch.cumsum(F.softplus(raw[..., 1:]), dim=-1)], dim=-1)


class MSSTNet(nn.Module):
    def __init__(self, config: MSSTNetConfig) -> None:
        super().__init__()
        self.config = config
        self.calendar_embedding = CalendarEmbedding(config)
        self.hourly_encoder = DualPathScaleEncoder(
            config, config.hourly_features, config.hourly_length,
            config.hourly_patch, config.hourly_length // config.hourly_patch,
            config.trend_windows[0], float(config.hourly_patch))
        self.daily_encoder = DualPathScaleEncoder(
            config, config.daily_features, config.daily_length,
            config.daily_patch, config.daily_patch_count,
            config.trend_windows[1], float(config.daily_patch * 24))
        self.weekly_encoder = DualPathScaleEncoder(
            config, config.weekly_features, config.weekly_length,
            config.weekly_patch, config.weekly_length // config.weekly_patch,
            config.trend_windows[2], float(config.weekly_patch * 168))
        self.cross_resolution = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
        self.cross_norm    = nn.LayerNorm(config.d_model)
        self.decoder       = FutureCovariateDecoder(config)
        self.point_head    = nn.Linear(config.d_model, 1)
        self.quantile_head = MonotoneQuantileHead(config.d_model, len(config.quantiles))

    def _cal(self, batch: Mapping[str, Tensor], prefix: str) -> Tensor:
        return self.calendar_embedding(
            batch[f"{prefix}_fourier"],
            batch[f"{prefix}_holiday_flags"],
            batch[f"{prefix}_holiday_type"],
        )

    def forward(self, batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        h_h = self.hourly_encoder(batch["hourly_values"].float(), self._cal(batch, "hourly"))
        h_d = self.daily_encoder( batch["daily_values"].float(),  self._cal(batch, "daily"))
        h_w = self.weekly_encoder(batch["weekly_values"].float(), self._cal(batch, "weekly"))
        coarse = torch.cat([h_d, h_w], dim=1)
        enriched, _ = self.cross_resolution(h_h, coarse, coarse, need_weights=False)
        h_h_enriched = self.cross_norm(h_h + enriched)
        ctx = torch.cat([h_h_enriched, h_d, h_w], dim=1)
        decoded   = self.decoder(self._cal(batch, "future"), ctx)
        point     = self.point_head(decoded).squeeze(-1)
        quantiles = self.quantile_head(decoded)
        return {"point": point, "quantiles": quantiles,
                "decoder_state": decoded, "encoder_context": ctx}


# ---------------------------------------------------------------------------
# Loss  (unchanged)
# ---------------------------------------------------------------------------

def estimate_transition_threshold(values: Sequence[float], quantile: float = 0.99) -> float:
    y = _as_1d_float(values)
    return float(np.quantile(np.abs(np.diff(y)), quantile)) if len(y) >= 2 else 0.0


def pinball_loss(y_true: Tensor, q_pred: Tensor, quantiles: Sequence[float]) -> Tensor:
    y    = y_true.unsqueeze(-1)
    taus = torch.tensor(quantiles, device=q_pred.device, dtype=q_pred.dtype).view(1, 1, -1)
    err  = y - q_pred
    return torch.maximum(taus * err, (taus - 1.0) * err).mean()


@dataclass
class MSSTLossConfig:
    quantile_weight: float = 0.30
    huber_delta: float = 1.0
    nonneg_weight: float = 0.0
    smoothness_weight: float = 0.0
    max_transition: Optional[float] = None


class MSSTLoss(nn.Module):
    def __init__(self, quantiles: Sequence[float], config: Optional[MSSTLossConfig] = None) -> None:
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        self.config    = config or MSSTLossConfig()

    def forward(self, outputs: Mapping[str, Tensor], target: Tensor) -> Dict[str, Tensor]:
        point     = outputs["point"]
        quantiles = outputs["quantiles"]
        huber  = F.huber_loss(point, target.float(), delta=self.config.huber_delta, reduction="mean")
        qloss  = pinball_loss(target.float(), quantiles, self.quantiles)
        total  = huber + self.config.quantile_weight * qloss
        nonneg = point.new_tensor(0.0)
        if self.config.nonneg_weight > 0:
            nonneg = torch.square(torch.clamp(-point, min=0.0)).mean()
            total  = total + self.config.nonneg_weight * nonneg
        smoothness = point.new_tensor(0.0)
        if self.config.smoothness_weight > 0 and self.config.max_transition is not None and point.size(1) > 1:
            excess     = torch.clamp(torch.abs(point[:, 1:] - point[:, :-1]) - float(self.config.max_transition), min=0.0)
            smoothness = torch.square(excess).mean()
            total      = total + self.config.smoothness_weight * smoothness
        return {"loss": total, "huber": huber.detach(), "pinball": qloss.detach(),
                "nonneg": nonneg.detach(), "smoothness": smoothness.detach()}


def move_batch_to_device(batch: Mapping[str, Tensor], device: torch.device | str) -> Dict[str, Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Performance metrics  (NEW)
# ---------------------------------------------------------------------------

def compute_metrics(
    point_np: np.ndarray,
    target_np: np.ndarray,
    quantiles_np: np.ndarray,
    quantile_levels: Sequence[float],
) -> Dict[str, float]:
    """
    point_np      : (N, H) point forecasts
    target_np     : (N, H) actual values
    quantiles_np  : (N, H, Q) quantile forecasts
    quantile_levels: e.g. [0.10, 0.25, 0.50, 0.75, 0.90]

    Returns
    -------
    mae      Mean Absolute Error (MWh)
    rmse     Root Mean Squared Error (MWh)
    mape     Mean Absolute Percentage Error (%)
    cov80    Coverage of [Q10, Q90] interval (target ≈ 80 %)
    winkler80 Winkler score for the 80 % interval (lower = better)
    """
    err   = point_np - target_np
    mae   = float(np.mean(np.abs(err)))
    rmse  = float(np.sqrt(np.mean(err ** 2)))
    mape  = float(np.mean(np.abs(err) / (np.abs(target_np) + 1e-8)) * 100)

    ql = list(quantile_levels)
    metrics: Dict[str, float] = {"mae": mae, "rmse": rmse, "mape": mape}

    # 80 % interval: Q10 → Q90
    if 0.10 in ql and 0.90 in ql:
        lo = quantiles_np[..., ql.index(0.10)]
        hi = quantiles_np[..., ql.index(0.90)]
        inside   = (target_np >= lo) & (target_np <= hi)
        cov80    = float(np.mean(inside) * 100)
        width    = hi - lo
        alpha    = 0.20  # 1 - 0.80
        penalty  = (2 / alpha) * np.where(target_np < lo, lo - target_np,
                   np.where(target_np > hi, target_np - hi, 0.0))
        winkler  = float(np.mean(width + penalty))
        metrics["cov80"]     = cov80
        metrics["winkler80"] = winkler

    return metrics


def print_metrics_guide() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              PERFORMANCE METRICS GUIDE                          ║
╠══════════════════════════════════════════════════════════════════╣
║ Point forecast                                                  ║
║  MAE      Mean Absolute Error (MWh)                            ║
║           Average unsigned deviation. Same unit as target.     ║
║           Lower is better. E.g. MAE=300 → avg ±300 MWh off.   ║
║                                                                 ║
║  RMSE     Root Mean Squared Error (MWh)                        ║
║           Like MAE but penalises large spikes harder.          ║
║           RMSE ≥ MAE always; big gap → occasional bad errors.  ║
║                                                                 ║
║  MAPE     Mean Absolute Percentage Error (%)                   ║
║           Scale-free. 3.2 % means avg forecast is 3.2 % off.  ║
║           Good energy benchmark: < 5 % is strong.             ║
║                                                                 ║
║  Huber    Smooth MAE/MSE blend (training loss term).           ║
║           Tracks with MAE but not directly interpretable.      ║
║                                                                 ║
║ Interval forecast (80 % interval = [Q10, Q90])                 ║
║  Pinball  Proper scoring rule for all quantile levels.         ║
║           Lower = better-calibrated quantile forecasts.        ║
║                                                                 ║
║  Cov80    % of actuals inside [Q10, Q90] interval.             ║
║           Target ≈ 80 %. Under → intervals too tight.          ║
║           Over  → intervals too wide (less useful).            ║
║                                                                 ║
║  Winkler  Interval width + penalty for misses (lower = better) ║
║           Rewards narrow intervals that still contain actuals. ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ---------------------------------------------------------------------------
# DDP training infrastructure  (unchanged)
# ---------------------------------------------------------------------------

def _setup_ddp(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup_ddp() -> None:
    dist.destroy_process_group()


def _train_epoch_amp(
    model: nn.Module, loader: DataLoader,
    optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
    loss_fn: MSSTLoss, device: torch.device, grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(batch)
            losses  = loss_fn(outputs, batch["target"])
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bsz = int(batch["target"].size(0))
        count += bsz
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu()) * bsz
    return {k: v / max(count, 1) for k, v in totals.items()}


@torch.no_grad()
def _validate_amp(
    model: nn.Module, loader: DataLoader,
    loss_fn: MSSTLoss, device: torch.device,
    quantile_levels: Sequence[float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Returns (loss_metrics, perf_metrics)."""
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    all_points:    List[np.ndarray] = []
    all_targets:   List[np.ndarray] = []
    all_quantiles: List[np.ndarray] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast():
            out   = model(batch)
            vloss = loss_fn(out, batch["target"])
        bsz = int(batch["target"].size(0))
        count += bsz
        for k, v in vloss.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu()) * bsz
        all_points.append(out["point"].detach().cpu().numpy())
        all_quantiles.append(out["quantiles"].detach().cpu().numpy())
        all_targets.append(batch["target"].detach().cpu().numpy())

    loss_metrics = {k: v / max(count, 1) for k, v in totals.items()}
    perf_metrics = compute_metrics(
        np.concatenate(all_points,    axis=0),
        np.concatenate(all_targets,   axis=0),
        np.concatenate(all_quantiles, axis=0),
        quantile_levels,
    )
    return loss_metrics, perf_metrics


def estimate_acf_initializers(values: Sequence[float]) -> Dict[str, float]:
    y     = _as_1d_float(values)
    y     = y - np.mean(y)
    denom = float(np.dot(y, y))
    if denom <= 0:
        return {"acf_lag1": 0.0, "acf_lag24": 0.0, "acf_lag168": 0.0}
    def acf(lag: int) -> float:
        return float(np.dot(y[lag:], y[:-lag]) / denom) if len(y) > lag else 0.0
    return {"acf_lag1": acf(1), "acf_lag24": acf(24), "acf_lag168": acf(168)}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_data_path(configured_path: str) -> str:
    """Return the first existing path for the target filename.

    Kaggle mounts datasets under /kaggle/input/<slug>/<file>.
    We search recursively so the cfg path only needs the filename to be correct.
    """
    import glob as _glob

    if os.path.exists(configured_path):
        return configured_path

    filename = os.path.basename(configured_path)
    # Search all Kaggle input subdirectories
    matches = _glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if matches:
        found = matches[0]
        print(f"[resolve] '{filename}' found at {found}")
        return found

    # Last resort: list what is available so user can update cfg
    available = _glob.glob("/kaggle/input/**/*", recursive=True)
    csv_files = [p for p in available if p.endswith((".csv", ".xlsx", ".xls"))]
    raise FileNotFoundError(
        f"Cannot find '{filename}' under /kaggle/input/.\n"
        f"Available data files:\n" + "\n".join(f"  {p}" for p in csv_files[:20])
    )


# Main worker
# ---------------------------------------------------------------------------

def _main_worker(rank: int, world_size: int, cfg: dict) -> None:
    _setup_ddp(rank, world_size)
    is_main = rank == 0
    device  = torch.device(f"cuda:{rank}")

    if is_main:
        print_metrics_guide()
        print("Loading and preprocessing dataset …")

    # ── Load & preprocess ────────────────────────────────────────────────
    path  = _resolve_data_path(cfg["data_path"])
    raw   = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df    = preprocess_rich_df(raw, cfg["timestamp_col"], target_col=cfg["target_col"])

    # ── Train/val split at the DataFrame level (time-ordered) ────────────
    n_rows  = len(df)
    n_train = int(n_rows * (1 - cfg["val_frac"]))
    df_train = df.iloc[:n_train].copy()
    df_val   = df.iloc[n_train:].copy()

    # ── Build datasets — val uses train's transformer & norm_stats ────────
    train_ds = RichMultiScaleEnergyDataset(
        df_train, cfg["timestamp_col"], cfg["target_col"],
        horizon=cfg["horizon"], stride=cfg["stride"],
    )
    val_ds = RichMultiScaleEnergyDataset(
        df_val, cfg["timestamp_col"], cfg["target_col"],
        horizon=cfg["horizon"], stride=cfg["stride"],
        transformer=train_ds.transformer,
        fit_transformer=False,
        norm_stats=train_ds.norm_stats,
    )

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                       shuffle=True, drop_last=True)
    train_loader  = DataLoader(train_ds, batch_size=cfg["batch_size"],
                               sampler=train_sampler, num_workers=cfg["num_workers"],
                               pin_memory=True)
    val_loader = (DataLoader(val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)
                  if is_main else None)

    # ── Model ─────────────────────────────────────────────────────────────
    finite_y = train_ds.transformed_y[np.isfinite(train_ds.transformed_y)]
    config   = MSSTNetConfig(
        horizon=cfg["horizon"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"],
        hourly_features=RichMultiScaleEnergyDataset.N_HOURLY_FEATURES,
        **estimate_acf_initializers(finite_y),
    )
    model   = MSSTNet(config).to(device)
    model   = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    loss_fn = MSSTLoss(
        config.quantiles,
        MSSTLossConfig(
            quantile_weight=0.30,
            huber_delta=float(np.std(train_ds.raw_y)),
            nonneg_weight=1e-4, smoothness_weight=1e-4,
            max_transition=estimate_transition_threshold(train_ds.raw_y),
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler    = torch.cuda.amp.GradScaler()

    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Device         : {world_size}×T4  (DDP + AMP)\n"
            f"Train samples  : {len(train_ds)}   Val samples: {len(val_ds)}\n"
            f"Hourly features: {config.hourly_features}  "
            f"(transformed + rolling_mean_24h + rolling_std_24h + diff_24h)\n"
            f"Params         : {n_params:,}\n"
            f"Eff. batch     : {cfg['batch_size'] * world_size}\n"
            f"fracdiff width : {int(train_ds.transformer.metadata_['fracdiff_width'])}\n"
        )
        hdr = (f"{'Ep':>3}  {'TrLoss':>8}  {'Huber':>8}  {'Pinball':>8}  "
               f"{'ValLoss':>8}  {'MAE':>7}  {'RMSE':>7}  {'MAPE%':>6}  "
               f"{'Cov80%':>7}  {'Winkler':>8}  {'LR':>9}")
        print(hdr)
        print("─" * len(hdr))

    best_val = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        train_sampler.set_epoch(epoch)
        tr = _train_epoch_amp(model, train_loader, optimizer, scaler, loss_fn, device)
        scheduler.step()
        dist.barrier()

        if is_main:
            vl, pm = _validate_amp(model, val_loader, loss_fn, device, config.quantiles)
            lr = scheduler.get_last_lr()[0]
            print(
                f"{epoch:>3}  {tr['loss']:>8.1f}  {tr['huber']:>8.1f}  {tr['pinball']:>8.2f}  "
                f"{vl['loss']:>8.1f}  {pm['mae']:>7.1f}  {pm['rmse']:>7.1f}  {pm['mape']:>6.2f}  "
                f"{pm.get('cov80', float('nan')):>7.1f}  {pm.get('winkler80', float('nan')):>8.1f}  "
                f"{lr:>9.2e}"
            )
            if vl["loss"] < best_val:
                best_val = vl["loss"]
                torch.save({
                    "epoch": epoch, "model_state": model.module.state_dict(),
                    "config": config, "transformer_meta": train_ds.transformer.metadata_,
                    "norm_stats": train_ds.norm_stats,
                    "val_metrics": {**vl, **pm},
                }, cfg["save_path"])

    if is_main:
        print(f"\nDone. Best val loss: {best_val:.2f}  →  {cfg['save_path']}")
    _cleanup_ddp()


# ---------------------------------------------------------------------------
# Config + entry point
# ---------------------------------------------------------------------------

cfg: dict = {
    # Excel file confirmed present at this path in Kaggle input.
    # rolling_mean_24h / rolling_std_24h / diff_24h are computed on-the-fly.
    "data_path":     "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx",
    "timestamp_col": "Start time UTC",
    "target_col":    "Electricity consumption (MWh)",
    "save_path":     "/kaggle/working/msst_net_v2_best.pt",

    "horizon":   24,
    "stride":    24,
    "val_frac":  0.15,

    "d_model":   128,
    "n_heads":   8,
    "n_layers":  3,
    "dropout":   0.10,

    "epochs":      30,
    "batch_size":  16,
    "lr":          1e-3,
    "num_workers": 2,
}

if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        import linecache, shutil, subprocess, sys, traceback
        cell_file   = traceback.extract_stack()[-1].filename
        script_path = "/kaggle/working/msst_net_v2_kaggle.py"
        lines = linecache.getlines(cell_file)
        if lines:
            with open(script_path, "w") as _f:
                _f.writelines(lines)
        else:
            shutil.copy(cell_file, script_path)
        print(f"Script written to {script_path}\nLaunching 2×T4 training via torchrun …\n")
        subprocess.run(
            [sys.executable, "-m", "torch.distributed.run",
             "--nproc_per_node=2", "--standalone", script_path],
            check=True,
        )
    else:
        rank       = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        _main_worker(rank, world_size, cfg)
