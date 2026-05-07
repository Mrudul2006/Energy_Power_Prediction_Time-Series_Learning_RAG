"""
Rolling-origin accuracy benchmark for hourly energy-consumption forecasting.

The benchmark starts with strong seasonal baselines and the repository's
BoostedTreesForecaster because those give a useful accuracy floor before
spending time on heavier neural models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

# Pandas treats these compiled accelerators as optional. Some local scientific
# stacks have NumPy 2 with old accelerator wheels, which creates noisy import
# traces even though pandas can run without them.
for _optional_accelerator in ("numexpr", "bottleneck"):
    sys.modules.setdefault(_optional_accelerator, None)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DATA_PATH = ROOT / "Energy Consumption Dataset.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "benchmark_outputs"
TARGET_COL = "Electricity consumption (MWh)"
TIME_COL = "Start time UTC"


@dataclass(frozen=True)
class Fold:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int


def load_energy_series(path: Path) -> pd.Series:
    """Load, align to hourly buckets, and lightly repair the electricity series."""

    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    missing = {TIME_COL, TARGET_COL}.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).sort_values(TIME_COL)
    df[TIME_COL] = df[TIME_COL].dt.tz_localize(None)

    off_hour = (df[TIME_COL].dt.minute != 0) | (df[TIME_COL].dt.second != 0) | (df[TIME_COL].dt.microsecond != 0)
    df["hour"] = df[TIME_COL].dt.round("h")
    duplicate_hours = int(df["hour"].duplicated().sum())
    if off_hour.any() or duplicate_hours:
        print(
            f"Hourly alignment: rounded {int(off_hour.sum())} off-hour rows; "
            f"averaged {duplicate_hours} duplicate hourly buckets."
        )

    series = df.groupby("hour", sort=True)[TARGET_COL].mean()
    series.index = series.index.tz_localize(None)

    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    series = series.reindex(full_index)
    series = series.interpolate(method="time").ffill().bfill()
    series.name = TARGET_COL
    return series.astype(float)


def make_folds(series: pd.Series, horizon: int, max_folds: int, min_train_size: int) -> List[Fold]:
    """Create chronological rolling-origin folds near the end of the series."""

    horizon = int(horizon)
    max_folds = int(max_folds)
    min_train_size = int(min_train_size)
    if horizon <= 0 or max_folds <= 0:
        raise ValueError("horizon and max_folds must be positive")
    if len(series) < min_train_size + horizon:
        raise ValueError("not enough observations for the requested benchmark")

    possible = (len(series) - min_train_size) // horizon
    n_folds = min(max_folds, possible)
    if n_folds <= 0:
        raise ValueError("min_train_size leaves no room for test folds")

    start_cutoff = len(series) - n_folds * horizon
    folds: List[Fold] = []
    for fold in range(n_folds):
        train_end_pos = start_cutoff + fold * horizon
        test_start_pos = train_end_pos
        test_end_pos = test_start_pos + horizon
        train = series.iloc[:train_end_pos]
        test = series.iloc[test_start_pos:test_end_pos]
        folds.append(
            Fold(
                fold=fold + 1,
                train_end=pd.Timestamp(train.index[-1]),
                test_start=pd.Timestamp(test.index[0]),
                test_end=pd.Timestamp(test.index[-1]),
                train_size=len(train),
                test_size=len(test),
            )
        )
    return folds


def _repeat_lag_forecast(history: Sequence[float], horizon: int, lag: int) -> np.ndarray:
    arr = np.asarray(history, dtype=float)
    if len(arr) >= lag:
        pattern = arr[-lag:]
    else:
        pattern = arr[-1:]
    reps = int(np.ceil(horizon / len(pattern)))
    return np.tile(pattern, reps)[:horizon].astype(float)


def baseline_forecasts(train: pd.Series, horizon: int) -> Dict[str, np.ndarray]:
    daily = _repeat_lag_forecast(train, horizon, lag=24)
    weekly = _repeat_lag_forecast(train, horizon, lag=168)
    return {
        "naive_last": np.repeat(float(train.iloc[-1]), horizon),
        "seasonal_24h": daily,
        "seasonal_168h": weekly,
        "seasonal_blend": 0.5 * daily + 0.5 * weekly,
    }


def mase_denominator(train: pd.Series, seasonal_period: int = 24) -> float:
    arr = train.to_numpy(dtype=float)
    if len(arr) <= seasonal_period:
        diffs = np.diff(arr)
    else:
        diffs = arr[seasonal_period:] - arr[:-seasonal_period]
    denom = float(np.mean(np.abs(diffs))) if len(diffs) else np.nan
    return denom if np.isfinite(denom) and denom > 1e-12 else np.nan


def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float], train: pd.Series) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    err = pred - y
    denom = (np.abs(y) + np.abs(pred)) / 2.0
    smape_terms = np.divide(np.abs(err), denom, out=np.zeros_like(err), where=denom > 1e-12)
    mase_denom = mase_denominator(train)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "smape": float(100.0 * np.mean(smape_terms)),
        "mase": float(np.mean(np.abs(err)) / mase_denom) if np.isfinite(mase_denom) else np.nan,
        "bias": float(np.mean(err)),
    }


def boosted_tree_forecast(train: pd.Series, horizon: int, model_type: str) -> np.ndarray:
    from models.boosted_trees import BoostedTreesConfig, BoostedTreesForecaster

    config = BoostedTreesConfig(
        model_type=model_type,
        mode="direct",
        horizon=horizon,
        lags=(1, 2, 3, 23, 24, 25, 48, 72, 167, 168, 169, 336, 504, 672),
        rolling_windows=(3, 6, 12, 24, 48, 168, 336),
        calendar=True,
        model_params=_lightweight_tree_params(model_type),
    )
    model = BoostedTreesForecaster(config)
    model.fit(train)
    return model.forecast(horizon).mean


def residual_boosted_tree_forecast(
    train: pd.Series,
    horizon: int,
    model_type: str,
    baseline_name: str = "seasonal_24h",
) -> np.ndarray:
    """Forecast seasonal residuals, then add them back to the baseline."""

    from sklearn.base import clone

    from models.boosted_trees import BoostedTreesConfig, BoostedTreesForecaster
    from models.common import lag_feature_row, supervised_lag_frame

    lags = (1, 2, 3, 23, 24, 25, 48, 72, 167, 168, 169, 336, 504, 672)
    rolling_windows = (3, 6, 12, 24, 48, 168, 336)
    config = BoostedTreesConfig(
        model_type=model_type,
        mode="direct",
        horizon=horizon,
        lags=lags,
        rolling_windows=rolling_windows,
        calendar=True,
        model_params=_lightweight_tree_params(model_type),
    )
    estimator_factory = BoostedTreesForecaster(config)
    base_estimator = estimator_factory._make_estimator()

    history = train.to_numpy(dtype=float)
    models = {}
    feature_columns: List[str] = []
    for step in range(1, int(horizon) + 1):
        X, target = supervised_lag_frame(
            train,
            lags=lags,
            rolling_windows=rolling_windows,
            horizon=step,
            calendar=True,
        )
        baseline_at_target = _origin_aligned_baseline(history, step, baseline_name)
        baseline_series = pd.Series(baseline_at_target, index=train.index, name="baseline")
        combined = pd.concat([X, target, baseline_series], axis=1, sort=False).dropna()
        y_residual = combined["target"] - combined["baseline"]
        X_train = combined.drop(columns=["target", "baseline"])
        model = clone(base_estimator)
        model.fit(X_train, y_residual)
        models[step] = model
        if not feature_columns:
            feature_columns = X_train.columns.tolist()

    baseline = baseline_forecasts(train, horizon)[baseline_name]
    residuals: List[float] = []
    for step, timestamp in enumerate(pd.date_range(train.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"), start=1):
        row = lag_feature_row(history, timestamp, lags, rolling_windows)
        row = row.reindex(columns=feature_columns, fill_value=0.0)
        residuals.append(float(np.asarray(models[step].predict(row), dtype=float).reshape(-1)[0]))
    return baseline + np.asarray(residuals, dtype=float)


def _origin_aligned_baseline(history: np.ndarray, horizon: int, baseline_name: str) -> np.ndarray:
    """Return baseline values for y[t+horizon], indexed by origin t."""

    def lagged(lag: int) -> np.ndarray:
        values = np.full(len(history), np.nan, dtype=float)
        source_positions = np.arange(len(history)) + int(horizon) - int(lag)
        valid = (source_positions >= 0) & (source_positions < len(history))
        values[valid] = history[source_positions[valid]]
        return values

    if baseline_name == "seasonal_24h":
        return lagged(24)
    if baseline_name == "seasonal_168h":
        return lagged(168)
    if baseline_name == "seasonal_blend":
        return 0.5 * lagged(24) + 0.5 * lagged(168)
    if baseline_name == "naive_last":
        return history.astype(float)
    raise ValueError(f"unknown baseline_name: {baseline_name}")


def _lightweight_tree_params(model_type: str) -> Dict[str, float | int | str]:
    """Keep the first benchmark usable on laptops while retaining strong features."""

    model_type = model_type.lower()
    if model_type == "lightgbm":
        return {"n_estimators": 500, "num_leaves": 48, "learning_rate": 0.04, "verbosity": -1}
    if model_type == "xgboost":
        return {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.04}
    return {"max_iter": 350, "learning_rate": 0.04, "max_leaf_nodes": 47}


def evaluate_model(
    model_name: str,
    forecast_fn: Callable[[pd.Series, int], np.ndarray],
    series: pd.Series,
    folds: Iterable[Fold],
    horizon: int,
) -> tuple[list[dict], list[pd.DataFrame]]:
    metric_rows: List[dict] = []
    forecast_frames: List[pd.DataFrame] = []

    for fold in folds:
        train = series.loc[: fold.train_end]
        test = series.loc[fold.test_start : fold.test_end]
        pred = np.asarray(forecast_fn(train, horizon), dtype=float)[: len(test)]
        metrics = compute_metrics(test.to_numpy(dtype=float), pred, train)
        metric_rows.append({"model": model_name, **asdict(fold), **metrics})
        forecast_frames.append(
            pd.DataFrame(
                {
                    "fold": fold.fold,
                    "model": model_name,
                    "timestamp": test.index,
                    "actual": test.to_numpy(dtype=float),
                    "prediction": pred,
                }
            )
        )
    return metric_rows, forecast_frames


def run_benchmark(args: argparse.Namespace) -> None:
    series = load_energy_series(Path(args.data))
    folds = make_folds(series, args.horizon, args.max_folds, args.min_train_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[dict] = []
    all_forecasts: List[pd.DataFrame] = []

    for model_name, forecast in baseline_forecasts(series.iloc[: folds[0].train_size], args.horizon).items():
        del forecast
        rows, frames = evaluate_model(
            model_name,
            lambda train, h, name=model_name: baseline_forecasts(train, h)[name],
            series,
            folds,
            args.horizon,
        )
        all_metrics.extend(rows)
        all_forecasts.extend(frames)

    if args.include_boosted and not args.skip_boosted:
        try:
            rows, frames = evaluate_model(
                f"boosted_trees_{args.tree_backend}",
                lambda train, h: boosted_tree_forecast(train, h, args.tree_backend),
                series,
                folds,
                args.horizon,
            )
            all_metrics.extend(rows)
            all_forecasts.extend(frames)
        except (ImportError, AttributeError) as exc:
            print(f"Skipping boosted_trees_{args.tree_backend}: {exc}")
        try:
            rows, frames = evaluate_model(
                f"residual_boosted_{args.tree_backend}_{args.residual_baseline}",
                lambda train, h: residual_boosted_tree_forecast(
                    train,
                    h,
                    args.tree_backend,
                    baseline_name=args.residual_baseline,
                ),
                series,
                folds,
                args.horizon,
            )
            all_metrics.extend(rows)
            all_forecasts.extend(frames)
        except (ImportError, AttributeError) as exc:
            print(f"Skipping residual_boosted_{args.tree_backend}: {exc}")

    metrics = pd.DataFrame(all_metrics)
    forecasts = pd.concat(all_forecasts, ignore_index=True)
    summary = (
        metrics.groupby("model")[["mae", "rmse", "smape", "mase", "bias"]]
        .mean()
        .sort_values("mae")
    )

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    forecasts.to_csv(output_dir / "forecasts.csv", index=False)
    (output_dir / "summary.json").write_text(summary.to_json(orient="index", indent=2), encoding="utf-8")

    print(f"Loaded {len(series):,} hourly observations from {series.index[0]} to {series.index[-1]}")
    print(f"Evaluated {len(folds)} fold(s), horizon={args.horizon}")
    print(summary.round(4).to_string())
    print(f"\nWrote benchmark outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to Energy Consumption Dataset.xlsx")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for benchmark outputs")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon per fold")
    parser.add_argument("--max-folds", type=int, default=3, help="Number of rolling-origin folds")
    parser.add_argument("--min-train-size", type=int, default=24 * 60, help="Minimum training observations")
    parser.add_argument("--tree-backend", choices=["sklearn", "lightgbm", "xgboost"], default="sklearn")
    parser.add_argument("--include-boosted", action="store_true", help="Also evaluate BoostedTreesForecaster")
    parser.add_argument(
        "--residual-baseline",
        choices=["naive_last", "seasonal_24h", "seasonal_168h", "seasonal_blend"],
        default="seasonal_24h",
        help="Baseline used by residual boosted trees",
    )
    parser.add_argument("--skip-boosted", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
