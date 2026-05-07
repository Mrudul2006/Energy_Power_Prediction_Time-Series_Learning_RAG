"""
MSST-Net: Multi-Scale Seasonal Transformer Network — Kaggle 2×T4 DDP edition.

Identical architecture to msst_net.py. Training infrastructure replaced with:
  - torch.distributed (NCCL backend) via mp.spawn
  - DistributedDataParallel (DDP)
  - torch.cuda.amp (GradScaler + autocast) for T4 Tensor Cores
  - rank-0-only validation, printing, and checkpointing

Run on Kaggle:  !python msst_net_kaggle.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# Statistical input transformation
# ---------------------------------------------------------------------------


def _as_1d_float(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("expected at least one finite value")
    return arr


def estimate_boxcox_lambda(values: Sequence[float]) -> Tuple[float, float]:
    """Estimate Box-Cox lambda by profile MLE and return (lambda, shift)."""
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
    """Estimate fractional differencing d using the local Whittle objective."""
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
    metadata = {
        "d": d_hat,
        "bandwidth": float(bandwidth),
        "objective": float(opt.fun),
        "success": float(bool(opt.success)),
    }
    return d_hat, metadata


def fractional_diff_weights(d: float, threshold: float = 1e-5, max_size: int = 4096) -> np.ndarray:
    """Fixed-width coefficients for (1 - B)^d."""
    weights = [1.0]
    for k in range(1, max_size):
        next_weight = -weights[-1] * (d - k + 1.0) / k
        weights.append(float(next_weight))
        if abs(next_weight) < threshold and k > 32:
            break
    return np.asarray(weights, dtype=float)


def fractional_difference_fixed_width(
    values: Sequence[float],
    d: float,
    threshold: float = 1e-5,
    max_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply fixed-width fractional differencing.

    The first width - 1 values are NaN because a full causal fractional window
    is not available there. Dataset construction skips those rows.
    """
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
    """Analytical input transform fitted before the neural model."""

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
            z, d=d, threshold=self.fracdiff_threshold, max_size=self.max_fracdiff_weights,
        )
        self.lam = lam
        self.shift = shift
        self.d = d
        self.fracdiff_weights_ = weights
        self.metadata_ = {
            "boxcox_lambda": float(lam),
            "boxcox_shift": float(shift),
            "local_whittle_d": float(d),
            "fracdiff_width": float(len(weights)),
            "first_valid_index": float(np.argmax(np.isfinite(transformed))),
            **whittle,
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
            z, d=float(self.d), threshold=self.fracdiff_threshold, max_size=self.max_fracdiff_weights,
        )
        return transformed

    def fit_transform(self, values: Sequence[float]) -> np.ndarray:
        return self.fit(values).transform(values)


# ---------------------------------------------------------------------------
# Calendar and holiday covariates
# ---------------------------------------------------------------------------


INDIAN_HOLIDAY_TYPES: Tuple[str, ...] = (
    "none",
    "republic_day",
    "holi",
    "good_friday",
    "eid_al_fitr",
    "ram_navami",
    "mahavir_jayanti",
    "buddha_purnima",
    "independence_day",
    "raksha_bandhan",
    "janmashtami",
    "gandhi_jayanti",
    "dussehra",
    "diwali",
    "christmas",
    "other",
)


DEFAULT_FIXED_INDIAN_HOLIDAYS: Mapping[Tuple[int, int], str] = {
    (1, 26): "republic_day",
    (8, 15): "independence_day",
    (10, 2): "gandhi_jayanti",
    (12, 25): "christmas",
}


class IndianHolidayCalendar:
    """
    Lightweight Indian holiday encoder.

    Fixed national holidays are built in. Movable holidays can be supplied as a
    mapping from date-like values to holiday type names, for example
    {"2026-11-08": "diwali"}. Unknown names are mapped to "other".
    """

    def __init__(
        self,
        extra_holidays: Optional[Mapping[object, str]] = None,
        type_names: Sequence[str] = INDIAN_HOLIDAY_TYPES,
    ) -> None:
        self.type_names = tuple(type_names)
        self.type_to_id = {name: i for i, name in enumerate(self.type_names)}
        self.none_id = self.type_to_id.get("none", 0)
        self.other_id = self.type_to_id.get("other", len(self.type_names) - 1)
        self.extra: Dict[pd.Timestamp, str] = {}
        for key, value in (extra_holidays or {}).items():
            self.extra[pd.Timestamp(key).normalize()] = str(value)

    @property
    def num_types(self) -> int:
        return len(self.type_names)

    def _type_for_date(self, ts: pd.Timestamp) -> str:
        day = pd.Timestamp(ts).normalize()
        if day in self.extra:
            return self.extra[day]
        return DEFAULT_FIXED_INDIAN_HOLIDAYS.get((day.month, day.day), "none")

    def type_id_for_date(self, ts: pd.Timestamp) -> int:
        name = self._type_for_date(ts)
        return self.type_to_id.get(name, self.other_id)

    def features(self, timestamps: Sequence[pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray]:
        idx = pd.DatetimeIndex(timestamps)
        type_ids = np.asarray([self.type_id_for_date(ts) for ts in idx], dtype=np.int64)
        holiday = (type_ids != self.none_id).astype(np.float32)
        eve_ids = np.asarray([self.type_id_for_date(ts + pd.Timedelta(days=1)) for ts in idx], dtype=np.int64)
        eve = (eve_ids != self.none_id).astype(np.float32)
        flags = np.column_stack([holiday, eve]).astype(np.float32)
        return flags, type_ids


def fourier_calendar_features(timestamps: Sequence[pd.Timestamp]) -> np.ndarray:
    """
    Fixed ED, EW, and EA Fourier basis.

    Dimensions: daily 6 + weekly 4 + annual 2 = 12.
    Holiday flags and learned holiday type embedding are appended by the neural
    CalendarEmbedding module, bringing the default model covariate dimension to
    17. This corrects the arithmetic inconsistency in the written spec.
    """
    idx = pd.DatetimeIndex(timestamps)
    hour_of_day = idx.hour.to_numpy(dtype=float) + idx.minute.to_numpy(dtype=float) / 60.0
    hour_of_week = idx.dayofweek.to_numpy(dtype=float) * 24.0 + hour_of_day
    hour_of_year = (idx.dayofyear.to_numpy(dtype=float) - 1.0) * 24.0 + hour_of_day

    cols: List[np.ndarray] = []
    for k in (1, 2, 3):
        cols.append(np.sin(2.0 * np.pi * k * hour_of_day / 24.0))
        cols.append(np.cos(2.0 * np.pi * k * hour_of_day / 24.0))
    for k in (1, 2):
        cols.append(np.sin(2.0 * np.pi * k * hour_of_week / 168.0))
        cols.append(np.cos(2.0 * np.pi * k * hour_of_week / 168.0))
    cols.append(np.sin(2.0 * np.pi * hour_of_year / 8760.0))
    cols.append(np.cos(2.0 * np.pi * hour_of_year / 8760.0))
    return np.column_stack(cols).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset construction
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
    return np.column_stack(
        [
            np.mean(arr, axis=1),
            np.max(arr, axis=1),
            np.min(arr, axis=1),
            np.std(arr, axis=1),
        ]
    ).astype(np.float32)


class MultiScaleEnergyDataset(Dataset):
    """
    Produces one MSST-Net training sample per forecast origin.

    Targets are raw energy values. Historical inputs are the Box-Cox plus
    fractional-differenced transformed values.
    """

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
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
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
        self.calendar = holiday_calendar or IndianHolidayCalendar()
        self.stride = max(1, int(stride))

        finite = np.flatnonzero(np.isfinite(self.transformed_y))
        if len(finite) == 0:
            raise ValueError("transformed series has no finite values")
        self.first_valid = int(finite[0])
        self.min_history = max(168, 365 * 24, 52 * 168)
        first_origin = max(self.first_valid + self.min_history, self.min_history)
        last_origin = len(self.raw_y) - self.horizon
        self.origins = list(range(first_origin, last_origin + 1, self.stride))
        if not self.origins:
            raise ValueError(
                "not enough hourly history for MSST-Net. Need at least "
                f"{self.min_history} transformed history hours plus horizon {self.horizon}."
            )

    def __len__(self) -> int:
        return len(self.origins)

    def _calendar_tensors(self, timestamps: Sequence[pd.Timestamp]) -> Dict[str, Tensor]:
        fourier = fourier_calendar_features(timestamps)
        flags, type_ids = self.calendar.features(timestamps)
        return {
            "fourier": torch.from_numpy(fourier),
            "holiday_flags": torch.from_numpy(flags),
            "holiday_type": torch.from_numpy(type_ids.astype(np.int64)),
        }

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        origin = self.origins[item]
        origin_ts = self.index[origin]
        x = self.transformed_y

        hourly_values = x[origin - 168 : origin, None].astype(np.float32)
        daily_window = x[origin - 365 * 24 : origin]
        weekly_window = x[origin - 52 * 168 : origin]
        if not np.isfinite(hourly_values).all() or not np.isfinite(daily_window).all() or not np.isfinite(weekly_window).all():
            raise RuntimeError("dataset origin includes non-finite transformed values")

        daily_values = _block_stats(daily_window, block_size=24, count=365)
        weekly_stats = _block_stats(weekly_window, block_size=168, count=52)

        weekly_end_times = [origin_ts - pd.Timedelta(hours=168 * (52 - k)) + pd.Timedelta(hours=167) for k in range(52)]
        weekly_doy = np.asarray(
            [pd.Timestamp(ts).dayofyear / 366.0 for ts in weekly_end_times],
            dtype=np.float32,
        )[:, None]
        weekly_values = np.concatenate([weekly_stats, weekly_doy], axis=1).astype(np.float32)

        hourly_times = self.index[origin - 168 : origin]
        daily_times = [origin_ts - pd.Timedelta(hours=24 * (365 - k)) + pd.Timedelta(hours=23) for k in range(365)]
        future_times = self.index[origin : origin + self.horizon]

        out: Dict[str, Tensor] = {
            "hourly_values": torch.from_numpy(hourly_values),
            "daily_values": torch.from_numpy(daily_values),
            "weekly_values": torch.from_numpy(weekly_values),
            "target": torch.from_numpy(self.raw_y[origin : origin + self.horizon].astype(np.float32)),
            "origin_index": torch.tensor(origin, dtype=torch.long),
        }

        for prefix, timestamps in (
            ("hourly", hourly_times),
            ("daily", daily_times),
            ("weekly", weekly_end_times),
            ("future", future_times),
        ):
            cal = self._calendar_tensors(timestamps)
            out[f"{prefix}_fourier"] = cal["fourier"]
            out[f"{prefix}_holiday_flags"] = cal["holiday_flags"]
            out[f"{prefix}_holiday_type"] = cal["holiday_type"]
        return out


# ---------------------------------------------------------------------------
# Neural architecture
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
    hourly_features: int = 1
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
        self.config = config
        self.holiday_embedding = nn.Embedding(config.holiday_type_count, config.holiday_embedding_dim)

    def forward(self, fourier: Tensor, flags: Tensor, holiday_type: Tensor) -> Tensor:
        type_emb = self.holiday_embedding(holiday_type.long())
        return torch.cat([fourier.float(), flags.float(), type_emb], dim=-1)


class MovingAverageDecomposer(nn.Module):
    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = max(1, int(window))

    def forward(self, values: Tensor) -> Tuple[Tensor, Tensor]:
        if self.window <= 1:
            return values, values.new_zeros(values.shape)
        x = values.transpose(1, 2)
        left = self.window // 2
        right = self.window - 1 - left
        padded = F.pad(x, (left, right), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.window, stride=1).transpose(1, 2)
        return trend, values - trend


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim: int, patch_size: int, patch_count: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_count = int(patch_count)
        self.proj = nn.Linear(input_dim * self.patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        needed = self.patch_size * self.patch_count
        if x.size(1) < needed:
            raise ValueError(f"sequence length {x.size(1)} is shorter than needed patch span {needed}")
        x = x[:, -needed:, :]
        bsz, _, dim = x.shape
        x = x.reshape(bsz, self.patch_count, self.patch_size * dim)
        return self.dropout(self.norm(self.proj(x)))


class PeriodicAttentionBias(nn.Module):
    def __init__(
        self,
        token_count: int,
        patch_span_hours: float,
        acf_lag1: float,
        acf_lag24: float,
        acf_lag168: float,
        tau_hours: float,
    ) -> None:
        super().__init__()
        self.token_count = int(token_count)
        self.patch_span_hours = float(patch_span_hours)
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau_hours, 1e-3)), dtype=torch.float32))
        self.w_local = nn.Parameter(torch.tensor(float(acf_lag1), dtype=torch.float32))
        self.w_daily = nn.Parameter(torch.tensor(float(acf_lag24), dtype=torch.float32))
        self.w_weekly = nn.Parameter(torch.tensor(float(acf_lag168), dtype=torch.float32))

        pos = torch.arange(self.token_count, dtype=torch.float32) * self.patch_span_hours
        delta = torch.abs(pos[:, None] - pos[None, :])
        self.register_buffer("delta_hours", delta, persistent=False)
        self.register_buffer("daily_mask", ((delta > 0) & (torch.remainder(delta, 24.0) < 1e-4)).float(), persistent=False)
        self.register_buffer("weekly_mask", ((delta > 0) & (torch.remainder(delta, 168.0) < 1e-4)).float(), persistent=False)

    def forward(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        delta = self.delta_hours.to(device=device, dtype=dtype)
        tau = torch.exp(self.log_tau).to(device=device, dtype=dtype).clamp_min(1e-3)
        local = self.w_local.to(device=device, dtype=dtype) * torch.exp(-delta / tau)
        daily = self.w_daily.to(device=device, dtype=dtype) * self.daily_mask.to(device=device, dtype=dtype)
        weekly = self.w_weekly.to(device=device, dtype=dtype) * self.weekly_mask.to(device=device, dtype=dtype)
        return local + daily + weekly


class BiasedTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        hidden = ff_multiplier * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: Tensor, attn_bias: Optional[Tensor] = None) -> Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_bias, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))


class BiasedTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_multiplier: int,
        dropout: float,
        bias: Optional[PeriodicAttentionBias] = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [BiasedTransformerLayer(d_model, n_heads, ff_multiplier, dropout) for _ in range(n_layers)]
        )
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        bias = None
        if self.bias is not None:
            bias = self.bias(x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, bias)
        return x


class DualPathScaleEncoder(nn.Module):
    def __init__(
        self,
        config: MSSTNetConfig,
        value_dim: int,
        raw_length: int,
        patch_size: int,
        patch_count: int,
        moving_average_window: int,
        patch_span_hours: float,
    ) -> None:
        super().__init__()
        self.config = config
        self.raw_length = raw_length
        self.decomposer = MovingAverageDecomposer(moving_average_window)
        token_input_dim = value_dim + config.calendar_dim
        self.trend_patch = PatchEmbedding(token_input_dim, patch_size, patch_count, config.d_model, config.dropout)
        self.seasonal_patch = PatchEmbedding(token_input_dim, patch_size, patch_count, config.d_model, config.dropout)
        self.trend_encoder = BiasedTransformerEncoder(
            config.d_model, config.n_heads, config.n_layers, config.ff_multiplier, config.dropout, bias=None,
        )
        seasonal_bias = PeriodicAttentionBias(
            token_count=patch_count,
            patch_span_hours=patch_span_hours,
            acf_lag1=config.acf_lag1,
            acf_lag24=config.acf_lag24,
            acf_lag168=config.acf_lag168,
            tau_hours=config.tau_hours,
        )
        self.seasonal_encoder = BiasedTransformerEncoder(
            config.d_model, config.n_heads, config.n_layers, config.ff_multiplier, config.dropout, bias=seasonal_bias,
        )
        self.gate = nn.Linear(config.d_model, config.d_model)
        self.out_norm = nn.LayerNorm(config.d_model)

    def forward(self, values: Tensor, calendar: Tensor) -> Tensor:
        trend_values, seasonal_values = self.decomposer(values)
        trend_input = torch.cat([trend_values, calendar], dim=-1)
        seasonal_input = torch.cat([seasonal_values, calendar], dim=-1)
        h_trend = self.trend_encoder(self.trend_patch(trend_input))
        h_seasonal = self.seasonal_encoder(self.seasonal_patch(seasonal_input))
        gate = torch.sigmoid(self.gate(h_trend))
        return self.out_norm(gate * h_trend + (1.0 - gate) * h_seasonal)


class FutureCovariateDecoder(nn.Module):
    def __init__(self, config: MSSTNetConfig) -> None:
        super().__init__()
        self.query_mlp = nn.Sequential(
            nn.Linear(config.calendar_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ff_multiplier * config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_multiplier * config.d_model, config.d_model),
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, future_calendar: Tensor, encoder_context: Tensor) -> Tensor:
        query = self.query_mlp(future_calendar)
        attn_out, _ = self.cross_attn(query, encoder_context, encoder_context, need_weights=False)
        x = self.norm1(query + self.dropout(attn_out))
        return self.norm2(x + self.dropout(self.ff(x)))


class MonotoneQuantileHead(nn.Module):
    """Linear quantile head with non-crossing outputs."""

    def __init__(self, d_model: int, n_quantiles: int) -> None:
        super().__init__()
        self.raw = nn.Linear(d_model, n_quantiles)

    def forward(self, x: Tensor) -> Tensor:
        raw = self.raw(x)
        first = raw[..., :1]
        if raw.size(-1) == 1:
            return first
        increments = F.softplus(raw[..., 1:])
        return torch.cat([first, first + torch.cumsum(increments, dim=-1)], dim=-1)


class MSSTNet(nn.Module):
    """Multi-Scale Seasonal Transformer Network."""

    def __init__(self, config: MSSTNetConfig) -> None:
        super().__init__()
        self.config = config
        self.calendar_embedding = CalendarEmbedding(config)
        self.hourly_encoder = DualPathScaleEncoder(
            config,
            value_dim=config.hourly_features,
            raw_length=config.hourly_length,
            patch_size=config.hourly_patch,
            patch_count=config.hourly_length // config.hourly_patch,
            moving_average_window=config.trend_windows[0],
            patch_span_hours=float(config.hourly_patch),
        )
        self.daily_encoder = DualPathScaleEncoder(
            config,
            value_dim=config.daily_features,
            raw_length=config.daily_length,
            patch_size=config.daily_patch,
            patch_count=config.daily_patch_count,
            moving_average_window=config.trend_windows[1],
            patch_span_hours=float(config.daily_patch * 24),
        )
        self.weekly_encoder = DualPathScaleEncoder(
            config,
            value_dim=config.weekly_features,
            raw_length=config.weekly_length,
            patch_size=config.weekly_patch,
            patch_count=config.weekly_length // config.weekly_patch,
            moving_average_window=config.trend_windows[2],
            patch_span_hours=float(config.weekly_patch * 168),
        )
        self.cross_resolution = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.decoder = FutureCovariateDecoder(config)
        self.point_head = nn.Linear(config.d_model, 1)
        self.quantile_head = MonotoneQuantileHead(config.d_model, len(config.quantiles))

    def _calendar_from_batch(self, batch: Mapping[str, Tensor], prefix: str) -> Tensor:
        return self.calendar_embedding(
            batch[f"{prefix}_fourier"],
            batch[f"{prefix}_holiday_flags"],
            batch[f"{prefix}_holiday_type"],
        )

    def forward(self, batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        hourly_calendar = self._calendar_from_batch(batch, "hourly")
        daily_calendar = self._calendar_from_batch(batch, "daily")
        weekly_calendar = self._calendar_from_batch(batch, "weekly")
        future_calendar = self._calendar_from_batch(batch, "future")

        h_hourly = self.hourly_encoder(batch["hourly_values"].float(), hourly_calendar)
        h_daily = self.daily_encoder(batch["daily_values"].float(), daily_calendar)
        h_weekly = self.weekly_encoder(batch["weekly_values"].float(), weekly_calendar)

        coarse = torch.cat([h_daily, h_weekly], dim=1)
        enriched, _ = self.cross_resolution(h_hourly, coarse, coarse, need_weights=False)
        h_hourly_enriched = self.cross_norm(h_hourly + enriched)
        encoder_context = torch.cat([h_hourly_enriched, h_daily, h_weekly], dim=1)

        decoded = self.decoder(future_calendar, encoder_context)
        point = self.point_head(decoded).squeeze(-1)
        quantiles = self.quantile_head(decoded)
        return {
            "point": point,
            "quantiles": quantiles,
            "decoder_state": decoded,
            "encoder_context": encoder_context,
        }


# ---------------------------------------------------------------------------
# Losses, constraints, and training helpers
# ---------------------------------------------------------------------------


def estimate_transition_threshold(values: Sequence[float], quantile: float = 0.99) -> float:
    y = _as_1d_float(values)
    if len(y) < 2:
        return 0.0
    return float(np.quantile(np.abs(np.diff(y)), quantile))


def estimate_huber_delta(errors: Sequence[float], quantile: float = 0.95) -> float:
    e = np.abs(_as_1d_float(errors))
    return float(max(np.quantile(e, quantile), 1e-6))


def pinball_loss(y_true: Tensor, q_pred: Tensor, quantiles: Sequence[float]) -> Tensor:
    y = y_true.unsqueeze(-1)
    taus = torch.tensor(quantiles, device=q_pred.device, dtype=q_pred.dtype).view(1, 1, -1)
    err = y - q_pred
    return torch.maximum(taus * err, (taus - 1.0) * err).mean()


@dataclass
class MSSTLossConfig:
    quantile_weight: float = 0.30
    huber_delta: float = 1.0
    nonneg_weight: float = 0.0
    conservation_weight: float = 0.0
    smoothness_weight: float = 0.0
    max_transition: Optional[float] = None


class MSSTLoss(nn.Module):
    def __init__(self, quantiles: Sequence[float], config: Optional[MSSTLossConfig] = None) -> None:
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        self.config = config or MSSTLossConfig()

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        target: Tensor,
        estimated_daily_total: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        point = outputs["point"]
        quantiles = outputs["quantiles"]
        huber = F.huber_loss(point, target.float(), delta=self.config.huber_delta, reduction="mean")
        qloss = pinball_loss(target.float(), quantiles, self.quantiles)
        total = huber + self.config.quantile_weight * qloss

        nonneg = point.new_tensor(0.0)
        if self.config.nonneg_weight > 0:
            nonneg = torch.square(torch.clamp(-point, min=0.0)).mean()
            total = total + self.config.nonneg_weight * nonneg

        conservation = point.new_tensor(0.0)
        if self.config.conservation_weight > 0 and estimated_daily_total is not None and point.size(1) >= 24:
            first_day = point[:, :24].sum(dim=1)
            daily = estimated_daily_total.to(point.device, point.dtype).reshape_as(first_day)
            conservation = torch.square(first_day - daily).mean()
            total = total + self.config.conservation_weight * conservation

        smoothness = point.new_tensor(0.0)
        if self.config.smoothness_weight > 0 and self.config.max_transition is not None and point.size(1) > 1:
            jumps = torch.abs(point[:, 1:] - point[:, :-1])
            excess = torch.clamp(jumps - float(self.config.max_transition), min=0.0)
            smoothness = torch.square(excess).mean()
            total = total + self.config.smoothness_weight * smoothness

        return {
            "loss": total,
            "huber": huber.detach(),
            "pinball": qloss.detach(),
            "nonneg": nonneg.detach(),
            "conservation": conservation.detach(),
            "smoothness": smoothness.detach(),
        }


def move_batch_to_device(batch: Mapping[str, Tensor], device: torch.device | str) -> Dict[str, Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


# ---------------------------------------------------------------------------
# TBATS-style daily total regularizer source
# ---------------------------------------------------------------------------


class TBATSLikeDailyTotalModel:
    """
    Fourier-trend hourly baseline used as the daily-total teacher.

    This is deliberately dependency-light. If a production TBATS implementation
    is available, pass its 24-hour integral forecasts directly to MSSTLoss
    instead of using this class.
    """

    def __init__(self, ridge: float = 1e-3, periods: Sequence[int] = (24, 168, 8760), harmonics: Sequence[int] = (3, 2, 1)) -> None:
        self.ridge = float(ridge)
        self.periods = tuple(int(p) for p in periods)
        self.harmonics = tuple(int(k) for k in harmonics)
        self.beta_: Optional[np.ndarray] = None
        self.start_: Optional[pd.Timestamp] = None
        self.time_scale_: float = 1.0

    def _design(self, timestamps: Sequence[pd.Timestamp]) -> np.ndarray:
        idx = pd.DatetimeIndex(timestamps)
        if self.start_ is None:
            self.start_ = idx[0]
        t = ((idx - self.start_) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
        cols = [np.ones(len(idx)), t / max(self.time_scale_, 1.0)]
        for period, kmax in zip(self.periods, self.harmonics):
            for k in range(1, kmax + 1):
                cols.append(np.sin(2.0 * np.pi * k * t / period))
                cols.append(np.cos(2.0 * np.pi * k * t / period))
        return np.column_stack(cols).astype(float)

    def fit(self, timestamps: Sequence[pd.Timestamp], values: Sequence[float]) -> "TBATSLikeDailyTotalModel":
        y = np.asarray(values, dtype=float).reshape(-1)
        self.time_scale_ = max(float(len(y)), 1.0)
        x = self._design(timestamps)
        reg = self.ridge * np.eye(x.shape[1])
        reg[0, 0] = 0.0
        self.beta_ = np.linalg.solve(x.T @ x + reg, x.T @ y)
        return self

    def predict_hourly(self, timestamps: Sequence[pd.Timestamp]) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("fit the baseline before predicting")
        return self._design(timestamps) @ self.beta_

    def predict_first_day_total(self, future_timestamps: Sequence[pd.Timestamp]) -> float:
        idx = pd.DatetimeIndex(future_timestamps)[:24]
        if len(idx) < 24:
            raise ValueError("need at least 24 future timestamps for daily total")
        return float(np.sum(self.predict_hourly(idx)))


# ---------------------------------------------------------------------------
# Hierarchical temporal reconciliation
# ---------------------------------------------------------------------------


def build_temporal_summing_matrix(horizon: int, aggregate_sizes: Sequence[int] = (1, 24, 168, 720)) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Build S mapping bottom-level hourly forecasts to all hierarchy levels.

    Returns (S, rows), where each row is (aggregate_size, block_number).
    """
    rows: List[np.ndarray] = []
    labels: List[Tuple[int, int]] = []
    for size in aggregate_sizes:
        if size <= 0:
            raise ValueError("aggregate sizes must be positive")
        blocks = horizon // size
        for block in range(blocks):
            row = np.zeros(horizon, dtype=float)
            row[block * size : (block + 1) * size] = 1.0
            rows.append(row)
            labels.append((int(size), int(block)))
    if not rows:
        raise ValueError("horizon is shorter than all aggregate sizes")
    return np.vstack(rows), labels


def shrinkage_covariance(errors: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    e = np.asarray(errors, dtype=float)
    if e.ndim != 2:
        raise ValueError("errors must be a 2D array")
    cov = np.cov(e, rowvar=False)
    diag = np.diag(np.diag(cov))
    return (1.0 - shrinkage) * cov + shrinkage * diag + 1e-8 * np.eye(cov.shape[0])


class TemporalMinTReconciler:
    """
    Minimum-trace temporal reconciliation.

    Fit with validation errors for all hierarchy rows, then reconcile base
    forecasts from all hierarchy rows into coherent hourly and aggregate
    forecasts.
    """

    def __init__(self, horizon: int, aggregate_sizes: Sequence[int] = (1, 24, 168, 720), shrinkage: float = 0.10) -> None:
        self.horizon = int(horizon)
        self.aggregate_sizes = tuple(int(s) for s in aggregate_sizes)
        self.shrinkage = float(shrinkage)
        self.S, self.labels = build_temporal_summing_matrix(self.horizon, self.aggregate_sizes)
        self.projection_: Optional[np.ndarray] = None

    def fit(self, base_errors_all_levels: np.ndarray) -> "TemporalMinTReconciler":
        cov = shrinkage_covariance(base_errors_all_levels, shrinkage=self.shrinkage)
        precision = np.linalg.pinv(cov)
        middle = np.linalg.pinv(self.S.T @ precision @ self.S)
        self.projection_ = self.S @ middle @ self.S.T @ precision
        return self

    def reconcile_all(self, base_forecasts_all_levels: np.ndarray) -> np.ndarray:
        if self.projection_ is None:
            raise RuntimeError("fit the reconciler before reconciling")
        y = np.asarray(base_forecasts_all_levels, dtype=float)
        if y.shape[-1] != self.S.shape[0]:
            raise ValueError(f"expected last dimension {self.S.shape[0]}, got {y.shape[-1]}")
        return y @ self.projection_.T

    def reconcile_hourly(self, base_forecasts_all_levels: np.ndarray) -> np.ndarray:
        reconciled_all = self.reconcile_all(base_forecasts_all_levels)
        hourly_rows = [i for i, label in enumerate(self.labels) if label[0] == 1]
        return reconciled_all[..., hourly_rows]

    def aggregate_hourly_base(self, hourly: np.ndarray) -> np.ndarray:
        y = np.asarray(hourly, dtype=float)
        return y @ self.S.T


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


class ConformalQuantileCalibrator:
    """Split conformal calibration for lower/upper quantile pairs."""

    def __init__(self, quantiles: Sequence[float], alpha: float = 0.10) -> None:
        self.quantiles = tuple(float(q) for q in quantiles)
        self.alpha = float(alpha)
        self.thresholds_: Dict[Tuple[float, float], float] = {}

    def _pair_indices(self) -> List[Tuple[int, int]]:
        pairs = []
        q_to_i = {round(q, 8): i for i, q in enumerate(self.quantiles)}
        for i, q in enumerate(self.quantiles):
            if q < 0.5:
                upper = round(1.0 - q, 8)
                if upper in q_to_i:
                    pairs.append((i, q_to_i[upper]))
        return pairs

    def fit(self, quantile_predictions: np.ndarray, y_true: np.ndarray) -> "ConformalQuantileCalibrator":
        qpred = np.asarray(quantile_predictions, dtype=float)
        y = np.asarray(y_true, dtype=float)
        if qpred.ndim != 3:
            raise ValueError("quantile_predictions must have shape [n, horizon, n_quantiles]")
        n = qpred.shape[0] * qpred.shape[1]
        rank = min(n, int(math.ceil((n + 1) * (1.0 - self.alpha))))
        quantile_level = rank / max(n, 1)
        for lo_i, hi_i in self._pair_indices():
            lo_q = self.quantiles[lo_i]
            hi_q = self.quantiles[hi_i]
            lower = qpred[..., lo_i]
            upper = qpred[..., hi_i]
            scores = np.maximum.reduce([lower - y, y - upper, np.zeros_like(y)]).reshape(-1)
            self.thresholds_[(lo_q, hi_q)] = float(np.quantile(scores, quantile_level, method="higher"))
        return self

    def apply(self, quantile_predictions: np.ndarray) -> np.ndarray:
        if not self.thresholds_:
            raise RuntimeError("fit the calibrator before applying it")
        out = np.asarray(quantile_predictions, dtype=float).copy()
        q_to_i = {round(q, 8): i for i, q in enumerate(self.quantiles)}
        for (lo_q, hi_q), threshold in self.thresholds_.items():
            out[..., q_to_i[round(lo_q, 8)]] -= threshold
            out[..., q_to_i[round(hi_q, 8)]] += threshold
        return out


# ---------------------------------------------------------------------------
# Model factory from data
# ---------------------------------------------------------------------------


def estimate_acf_initializers(values: Sequence[float]) -> Dict[str, float]:
    y = _as_1d_float(values)
    y = y - np.mean(y)
    denom = float(np.dot(y, y))
    if denom <= 0:
        return {"acf_lag1": 0.0, "acf_lag24": 0.0, "acf_lag168": 0.0}

    def acf(lag: int) -> float:
        if len(y) <= lag:
            return 0.0
        return float(np.dot(y[lag:], y[:-lag]) / denom)

    return {
        "acf_lag1": acf(1),
        "acf_lag24": acf(24),
        "acf_lag168": acf(168),
    }


# ---------------------------------------------------------------------------
# DDP training infrastructure
# ---------------------------------------------------------------------------


def _setup_ddp(rank: int, world_size: int) -> None:
    # torchrun already sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE.
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    # Guard against stale process group left by a previous failed run.
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup_ddp() -> None:
    dist.destroy_process_group()


def _train_epoch_amp(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loss_fn: MSSTLoss,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(batch)
            losses = loss_fn(outputs, batch["target"])
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bsz = int(batch["target"].size(0))
        count += bsz
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * bsz
    return {k: v / max(count, 1) for k, v in totals.items()}


@torch.no_grad()
def _validate_amp(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: MSSTLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast():
            out = model(batch)
            vloss = loss_fn(out, batch["target"])
        bsz = int(batch["target"].size(0))
        count += bsz
        for k, v in vloss.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu()) * bsz
    return {k: v / max(count, 1) for k, v in totals.items()}


def _main_worker(rank: int, world_size: int, cfg: dict) -> None:
    _setup_ddp(rank, world_size)
    is_main = rank == 0
    device = torch.device(f"cuda:{rank}")

    # ── Dataset (built once per process; transformer is fitted on rank 0 implicitly) ──
    if is_main:
        print("Loading and preparing dataset …")

    path = cfg["data_path"]
    frame = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    dataset = MultiScaleEnergyDataset(
        frame,
        timestamp_col=cfg["timestamp_col"],
        target_col=cfg["target_col"],
        horizon=cfg["horizon"],
        stride=cfg["stride"],
    )

    n_total = len(dataset)
    n_val   = max(1, int(n_total * cfg["val_frac"]))
    n_train = n_total - n_val
    train_ds = Subset(dataset, list(range(n_train)))
    val_ds   = Subset(dataset, list(range(n_train, n_total)))

    # Each rank gets a non-overlapping shard of the training data
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader  = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    # Validation runs only on rank 0
    val_loader = (
        DataLoader(val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                   num_workers=cfg["num_workers"], pin_memory=True)
        if is_main else None
    )

    # ── Model ────────────────────────────────────────────────────────────────
    finite_y = dataset.transformed_y[np.isfinite(dataset.transformed_y)]
    config = MSSTNetConfig(
        horizon=cfg["horizon"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        **estimate_acf_initializers(finite_y),
    )
    model = MSSTNet(config).to(device)
    # find_unused_parameters=False is fine because all parameters are used
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    loss_fn   = MSSTLoss(
        config.quantiles,
        MSSTLossConfig(
            quantile_weight=0.30,
            huber_delta=float(np.std(dataset.raw_y)),
            nonneg_weight=1e-4,
            smoothness_weight=1e-4,
            max_transition=estimate_transition_threshold(dataset.raw_y),
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler    = torch.cuda.amp.GradScaler()

    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\nDevice  : {world_size}×T4  (DDP + AMP)\n"
            f"Samples : {n_total}  (train={n_train}, val={n_val})\n"
            f"Params  : {n_params:,}\n"
            f"Effective batch : {cfg['batch_size'] * world_size}  "
            f"({cfg['batch_size']} per GPU × {world_size} GPUs)\n"
            f"fracdiff width  : {int(dataset.transformer.metadata_['fracdiff_width'])}\n"
        )
        header = f"{'Epoch':>5}  {'Train':>10}  {'Huber':>10}  {'Pinball':>10}  {'Val':>10}  {'LR':>10}"
        print(header)
        print("─" * len(header))

    best_val = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        train_sampler.set_epoch(epoch)  # ensures different shuffles per epoch
        train_metrics = _train_epoch_amp(model, train_loader, optimizer, scaler, loss_fn, device)
        scheduler.step()

        # Synchronise all ranks before validation so rank 0 has the latest weights
        dist.barrier()

        if is_main:
            val_metrics = _validate_amp(model, val_loader, loss_fn, device)
            current_lr  = scheduler.get_last_lr()[0]
            print(
                f"{epoch:>5}  "
                f"{train_metrics['loss']:>10.2f}  "
                f"{train_metrics['huber']:>10.2f}  "
                f"{train_metrics['pinball']:>10.2f}  "
                f"{val_metrics['loss']:>10.2f}  "
                f"{current_lr:>10.2e}"
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.module.state_dict(),  # unwrap DDP before saving
                        "config": config,
                        "transformer_meta": dataset.transformer.metadata_,
                    },
                    cfg["save_path"],
                )

    if is_main:
        print(f"\nDone. Best val loss: {best_val:.2f}  →  {cfg['save_path']}")

    _cleanup_ddp()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Launch from a Kaggle notebook cell with:
#
#   import subprocess, sys
#   subprocess.run([
#       sys.executable, "-m", "torch.distributed.run",
#       "--nproc_per_node=2", "--standalone",
#       "/kaggle/working/msst_net_kaggle.py",
#   ], check=True)
#
# torchrun spawns two processes and injects LOCAL_RANK / WORLD_SIZE as env
# vars, so each process calls _main_worker directly — no mp.spawn needed
# (mp.spawn breaks in Jupyter kernels because __main__ is a frozen importer).
# ---------------------------------------------------------------------------

cfg: dict = {
    # ── Paths ─────────────────────────────────────────────────────────────
    "data_path":      "/kaggle/input/datasets/raunak45/energy/Energy Consumption Dataset.xlsx",
    "timestamp_col":  "Start time UTC",
    "target_col":     "Electricity consumption (MWh)",
    "save_path":      "/kaggle/working/msst_net_best.pt",

    # ── Data ──────────────────────────────────────────────────────────────
    "horizon":    24,
    "stride":     24,    # one forecast origin per day
    "val_frac":   0.15,  # last 15 % of origins held out for validation

    # ── Model ─────────────────────────────────────────────────────────────
    "d_model":  128,
    "n_heads":  8,
    "n_layers": 3,
    "dropout":  0.10,

    # ── Training ──────────────────────────────────────────────────────────
    "epochs":      30,
    "batch_size":  16,       # 16 per GPU × 2 GPUs = 32 effective batch
    "lr":          1e-3,
    "num_workers": 2,
}

if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        # Running as a notebook cell (papermill / interactive Jupyter).
        # The cell source lives in linecache even when no .py file exists on disk.
        import linecache, shutil, subprocess, sys, traceback
        cell_file   = traceback.extract_stack()[-1].filename
        script_path = "/kaggle/working/msst_net_kaggle.py"
        lines = linecache.getlines(cell_file)
        if lines:
            with open(script_path, "w") as _f:
                _f.writelines(lines)
        else:
            # Genuine file on disk (e.g. python msst_net_kaggle.py)
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
