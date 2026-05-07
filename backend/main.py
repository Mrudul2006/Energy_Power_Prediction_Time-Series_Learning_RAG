from __future__ import annotations

import math
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

BASE_DIR = Path(__file__).resolve().parent.parent

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow `from explainer import ...` whether main.py is run as a module or as a script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from explainer import ExplainerService, ExplainContext  # noqa: E402
from online_learner import OnlineLearner  # noqa: E402


# ── constants ─────────────────────────────────────────────────────────────────

BUFFER_HOURS = 5


# ── pydantic models ───────────────────────────────────────────────────────────

class HourObservation(BaseModel):
    timestamp: str
    actual: float = Field(gt=0)


class ForecastRequest(BaseModel):
    history: list[HourObservation] = Field(default_factory=list)
    bufferHours: int = BUFFER_HOURS


class ForecastPoint(BaseModel):
    timestamp: str
    actual: float | None
    predicted: float
    predicted_online: float | None = None
    online_correction: float | None = None
    lower: float
    upper: float
    error: float | None
    insight: str
    isForecast: bool


class ForecastMetrics(BaseModel):
    mae: float
    rmse: float
    mse: float
    mape: float


class ForecastResponse(BaseModel):
    metrics: ForecastMetrics
    series: list[ForecastPoint]
    recommendations: list[str]
    source: str = "api"


# ── global dataset cache ──────────────────────────────────────────────────────

_DATASET: dict[str, float] | None = None

def _load_dataset():
    global _DATASET
    if _DATASET is not None:
        return _DATASET
    
    import pandas as pd
    try:
        csv_path = BASE_DIR / "model_learning" / "datasets" / "cleaned_energy_data_model2.csv"
        if not csv_path.exists():
            csv_path = BASE_DIR / "model_learning" / "datasets" / "cleaned_energy_data.csv"
            
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Assuming columns: start_time and consumption
            _DATASET = {}
            for _, row in df.iterrows():
                try:
                    # Safely handle the user's specific DD-MM-YYYY HH:MM format properly
                    dt = pd.to_datetime(row['start_time'], dayfirst=True)
                    dt_str = dt.strftime("%Y-%m-%dT%H:00")
                    _DATASET[dt_str] = float(row['consumption'])
                except Exception:
                    pass
        else:
            _DATASET = {}
    except Exception as e:
        print("Error loading dataset:", e)
        _DATASET = {}
    return _DATASET


# ── app setup ─────────────────────────────────────────────────────────────────

_explainer: ExplainerService | None = None
_online_learner: OnlineLearner = OnlineLearner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _explainer, _online_learner
    _explainer = await ExplainerService.create()
    _online_learner = OnlineLearner.load()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:3002", "http://127.0.0.1:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


# Load specific models
model_lgbm = lgb.Booster(model_file=str(BASE_DIR / "model_learning" / "trained_models" / "final_lgbm.txt"))
model_xgb = xgb.XGBRegressor()
model_xgb.load_model(str(BASE_DIR / "model_learning" / "trained_models" / "final_xgb.json"))
model_cat = CatBoostRegressor()
model_cat.load_model(str(BASE_DIR / "model_learning" / "trained_models" / "final_cat.cbm"))


# ── forecast endpoint (unchanged logic) ──────────────────────────────────────

@app.get("/api/history")
def get_history(start: str, end: str):
    """
    Returns actual observed values from the dataset between start and end dates.
    """
    ds = _load_dataset()
    import pandas as pd
    try:
        start_dt = pd.to_datetime(start)
        start_time = datetime(start_dt.year, start_dt.month, start_dt.day, start_dt.hour)
        end_dt = pd.to_datetime(end)
        end_time = datetime(end_dt.year, end_dt.month, end_dt.day, end_dt.hour)
        print(start_time)
    except Exception:
        return {"history": []}
    
    if end_time < start_time:
        start_time, end_time = end_time, start_time

    history_points = []
    current_time = start_time
    while current_time <= end_time:
        t_str = current_time.strftime("%Y-%m-%dT%H:00")
        if t_str in ds:
            history_points.append({
                "timestamp": to_hour_value(current_time),
                "actual": ds[t_str]
            })
        current_time += timedelta(hours=1)
            
    return {"history": history_points}


@app.get("/api/dataset")
def get_dataset(limit: int = 1000):
    """
    Returns the raw dataset points. Useful for the frontend to know what dates are valid.
    Change limit to 0 to return the entire dataset.
    """
    ds = _load_dataset()
    points = [{"timestamp": k, "actual": v} for k, v in ds.items()]
    
    if limit > 0:
        points = points[-limit:]
        
    return {"dataset": points}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded"}


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    global _online_learner
    history = sorted(request.history, key=lambda item: parse_hour(item.timestamp))
    buffer_hours = BUFFER_HOURS

    # Build a fresh session learner that replays history sequentially
    session_learner = OnlineLearner()
    session_learner.buffer_X = list(_online_learner.buffer_X)
    session_learner.buffer_y = list(_online_learner.buffer_y)
    session_learner._ema     = _online_learner._ema
    session_learner._n       = _online_learner._n
    session_learner._ridge   = _online_learner._ridge

    observed_points = []
    for i in range(len(history)):
        item = history[i]
        previous = history[:i]
        ts = parse_hour(item.timestamp)
        base_pred = round(predict_value(previous, ts, fallback=item.actual), 2)
        features  = make_features(previous, ts) if previous else make_features(history[:1], ts)
        correction = round(session_learner.predict_correction(features), 2)
        online_pred = round(base_pred + correction, 2)
        error = round(item.actual - base_pred, 2)
        error_pct = round(abs(error) / max(1.0, item.actual) * 100, 2)
        cond  = condition_from_error(error_pct)
        lower, upper = prediction_interval(previous, base_pred)
        observed_points.append(dict(
            timestamp=to_hour_value(ts),
            actual=round(item.actual, 2),
            predicted=base_pred,
            predicted_online=online_pred,
            online_correction=correction,
            lower=lower, upper=upper,
            error=error,
            condition=cond,
            insight=build_insight(ts, item.actual, base_pred, cond, False),
            isForecast=False,
        ))
        # Update session learner with the residual AFTER prediction
        session_learner.update(features, error, save_to_disk=False)

    # Persist updated learner
    _online_learner = session_learner
    _online_learner.save()

    future_points = build_future_points_online(history, buffer_hours, session_learner)
    metrics = calculate_metrics(observed_points)

    return ForecastResponse(
        metrics=metrics,
        series=[*observed_points[-24:], *future_points],
        recommendations=build_recommendations(history, future_points, metrics),
        source="api",
    )


# ── explain endpoint ──────────────────────────────────────────────────────────

@app.post("/explain")
async def explain(forecast: ForecastResponse) -> dict:
    """
    Accept a ForecastResponse, build an ExplainContext from it, and return
    LLM-generated insights for the dashboard AI panel.

    The explainer auto-selects the best available backend:
      2. Ollama (localhost:11434 or OLLAMA_BASE_URL)
      3. Template fallback (always available)
    """
    global _explainer
    if _explainer is None:
        _explainer = await ExplainerService.create()

    ctx = _build_explain_context(forecast)
    return await _explainer.explain("online_forecast", ctx)


# ── 2021 full-year forecast endpoint ─────────────────────────────────────────

@app.get("/api/2021")
def get_2021_forecast(mode: str = "base", resolution: str = "daily"):
    """
    Returns 2021 actual vs ensemble predictions.
    mode: 'base' | 'online_replay'
    resolution: 'hourly' | 'daily' | 'weekly'
    """
    import pandas as pd

    FEATURE_COLS = [
        "lag_24h", "lag_48h", "lag_72h", "lag_168h",
        "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h",
        "ewm_24h", "hour", "day_of_week", "month", "year",
        "season", "is_weekend", "prev_daily_mean", "prev_daily_max",
    ]

    csv_path = BASE_DIR / "model_learning" / "datasets" / "cleaned_energy_data_model2.csv"
    df = pd.read_csv(csv_path)
    df["dt"] = pd.to_datetime(df["start_time"], dayfirst=False, format="mixed")
    df2021 = df[df["dt"].dt.year == 2021].copy().sort_values("dt").reset_index(drop=True)

    df2021["quarter"] = df2021["dt"].dt.quarter
    df2021["day_of_year"] = df2021["dt"].dt.dayofyear

    # Trigonometric features
    import numpy as np
    hour = df2021["hour"]
    day_of_week = df2021["day_of_week"]
    
    df2021["sin_hour_1"] = np.sin(2 * np.pi * hour / 24)
    df2021["cos_hour_1"] = np.cos(2 * np.pi * hour / 24)
    df2021["sin_hour_2"] = np.sin(4 * np.pi * hour / 24)
    df2021["cos_hour_2"] = np.cos(4 * np.pi * hour / 24)
    df2021["sin_hour_3"] = np.sin(6 * np.pi * hour / 24)
    df2021["cos_hour_3"] = np.cos(6 * np.pi * hour / 24)

    df2021["sin_dow_1"] = np.sin(2 * np.pi * day_of_week / 7)
    df2021["cos_dow_1"] = np.cos(2 * np.pi * day_of_week / 7)
    df2021["sin_dow_2"] = np.sin(4 * np.pi * day_of_week / 7)
    df2021["cos_dow_2"] = np.cos(4 * np.pi * day_of_week / 7)
    df2021["sin_dow_3"] = np.sin(6 * np.pi * day_of_week / 7)
    df2021["cos_dow_3"] = np.cos(6 * np.pi * day_of_week / 7)

    FEATURE_COLS = [
        "hour", "day_of_week", "month", "is_weekend", "quarter", "day_of_year",
        "lag_24h", "lag_48h", "lag_72h", "lag_168h",
        "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h", "ewm_24h",
        "sin_hour_1", "cos_hour_1", "sin_hour_2", "cos_hour_2", "sin_hour_3", "cos_hour_3",
        "sin_dow_1", "cos_dow_1", "sin_dow_2", "cos_dow_2", "sin_dow_3", "cos_dow_3",
        "prev_daily_mean", "prev_daily_max"
    ]

    for col in FEATURE_COLS:
        if col not in df2021.columns:
            df2021[col] = 0.0
    df2021[FEATURE_COLS] = df2021[FEATURE_COLS].fillna(0.0)

    X = df2021[FEATURE_COLS].values
    actuals = df2021["consumption"].values.astype(float)
    timestamps = df2021["dt"].dt.strftime("%Y-%m-%dT%H:00").tolist()

    # Base ensemble predictions - Vectorized
    try:
        xp = model_xgb.predict(X)
        lp = model_lgbm.predict(X)
        cp = model_cat.predict(X)
        base_preds = (xp + lp + cp) / 3.0
    except Exception as e:
        print("Vectorized predict failed:", e)
        base_preds = actuals

    # Online replay (sequential simulation)
    online_preds = None
    if mode == "online_replay":
        replay = OnlineLearner()
        online_arr = np.zeros(len(X))
        for i, feat in enumerate(X):
            corr = replay.predict_correction(list(feat))
            online_arr[i] = base_preds[i] + corr
            replay.update(list(feat), float(actuals[i]) - base_preds[i], save_to_disk=False, refit=(i % 24 == 0))
        online_preds = online_arr

    # Build series
    series = [
        {
            "timestamp": timestamps[i],
            "actual": round(float(actuals[i]), 1),
            "predicted_base": round(float(base_preds[i]), 1),
            "predicted_online": round(float(online_preds[i]), 1) if online_preds is not None else None,
        }
        for i in range(len(timestamps))
    ]

    # Downsample
    if resolution == "daily":
        series = _downsample(series, 24)
    elif resolution == "weekly":
        series = _downsample(series, 168)

    def _metrics(act, pred):
        errs = np.abs(np.array(act) - np.array(pred))
        pcts = errs / np.maximum(1, np.array(act)) * 100
        mse  = float(np.mean(errs ** 2))
        return {"mae": round(float(np.mean(errs)), 1),
                "rmse": round(float(np.sqrt(mse)), 1),
                "mape": round(float(np.mean(pcts)), 2)}

    act_list  = [p["actual"] for p in series]
    base_list = [p["predicted_base"] for p in series]
    metrics_base = _metrics(act_list, base_list)
    metrics_online = None
    if online_preds is not None:
        metrics_online = _metrics(act_list, [p["predicted_online"] for p in series])

    return {
        "series": series,
        "metrics_base": metrics_base,
        "metrics_online": metrics_online,
        "mode": mode,
        "resolution": resolution,
        "total_hours": int(len(timestamps)),
    }


def _downsample(series: list[dict], window: int) -> list[dict]:
    """Average every `window` consecutive points."""
    out = []
    for i in range(0, len(series), window):
        chunk = series[i: i + window]
        if not chunk:
            continue
        avg = lambda key: round(sum(p[key] for p in chunk if p[key] is not None) / len(chunk), 1)
        out.append({
            "timestamp": chunk[0]["timestamp"],
            "actual": avg("actual"),
            "predicted_base": avg("predicted_base"),
            "predicted_online": avg("predicted_online") if chunk[0]["predicted_online"] is not None else None,
        })
    return out


def _build_explain_context(response: ForecastResponse) -> ExplainContext:
    series = response.series
    observed = [p for p in series if not p.isForecast]
    forecast = [p for p in series if p.isForecast]

    current_mw = float(observed[-1].actual or 0) if observed else 0.0

    peak   = max(forecast, key=lambda p: p.predicted) if forecast else None
    trough = min(forecast, key=lambda p: p.predicted) if forecast else None
    avg_mw = float(np.mean([p.predicted for p in forecast])) if forecast else current_mw

    trend_pct = 0.0
    if len(forecast) >= 2 and forecast[0].predicted:
        trend_pct = (forecast[-1].predicted - forecast[0].predicted) / max(1.0, abs(forecast[0].predicted)) * 100

    anomalies = sorted(
        [p for p in observed if p.error is not None and p.actual is not None],
        key=lambda p: abs(p.error or 0),
        reverse=True
    )[:3]

    return ExplainContext(
        model_name="online_forecast",
        display_name="EnergyCast Online",
        mae=response.metrics.mae,
        rmse=response.metrics.rmse,
        mape=response.metrics.mape,
        online_updates=len(observed),
        forecast_peak_mw=peak.predicted if peak else avg_mw,
        forecast_peak_time=peak.timestamp if peak else "N/A",
        forecast_trough_mw=trough.predicted if trough else avg_mw,
        forecast_trough_time=trough.timestamp if trough else "N/A",
        forecast_avg_mw=avg_mw,
        forecast_trend_pct=trend_pct,
        current_mw=current_mw,
        top_anomalies=[
            {
                "time":      p.timestamp,
                "actual":    p.actual or 0.0,
                "predicted": p.predicted,
                "deviation": round(
                    (p.actual - p.predicted) / max(1.0, abs(p.predicted)) * 100, 1
                ) if p.actual else 0.0,
            }
            for p in anomalies
        ],
    )


# ── forecast helpers ─────────────────────────────────────────────────────────

def build_observed_point(history: list[HourObservation], index: int):
    """Legacy helper used by build_recommendations alert_count. No online correction."""
    item = history[index]
    previous = history[:index]
    predicted = round(predict_value(previous, parse_hour(item.timestamp), fallback=item.actual), 2)
    error = round(item.actual - predicted, 2)
    error_percent = round(abs(error) / max(1.0, item.actual) * 100, 2)
    condition = condition_from_error(error_percent)
    lower, upper = prediction_interval(previous, predicted)
    return dict(
        timestamp=to_hour_value(parse_hour(item.timestamp)),
        actual=round(item.actual, 2),
        predicted=predicted,
        lower=lower, upper=upper,
        error=error,
        condition=condition,
        insight=build_insight(parse_hour(item.timestamp), item.actual, predicted, condition, False),
        isForecast=False,
    )


def build_future_points_online(
    history: list[HourObservation],
    buffer_hours: int,
    learner: OnlineLearner,
) -> list[dict]:
    if not history:
        return []
    future = []
    working = list(history)
    latest_time = parse_hour(working[-1].timestamp)
    for step in range(1, buffer_hours + 1):
        ts = latest_time + timedelta(hours=step)
        base_pred = round(predict_value(working, ts), 2)
        features  = make_features(working, ts)
        correction = round(learner.predict_correction(features), 2)
        online_pred = round(base_pred + correction, 2)
        lower, upper = prediction_interval(working, base_pred)
        cond = condition_from_forecast(base_pred, working)
        future.append(dict(
            timestamp=to_hour_value(ts),
            actual=None,
            predicted=base_pred,
            predicted_online=online_pred,
            online_correction=correction,
            lower=lower, upper=upper,
            error=None,
            condition=cond,
            insight=build_insight(ts, None, base_pred, cond, True),
            isForecast=True,
        ))
        working.append(HourObservation(timestamp=to_hour_value(ts), actual=online_pred))
    return future


def build_future_points(history: list[HourObservation], buffer_hours: int):
    """Compatibility wrapper – no online correction."""
    return build_future_points_online(history, buffer_hours, OnlineLearner())


def predict_value(history: list[HourObservation], timestamp: datetime, fallback: float | None = None) -> float:
    if history:
        features = make_features(history, timestamp)
        try:
            # Create a 2D array since the models expect (n_samples, n_features)
            X = np.asarray([features])
            
            pred_xgb = float(model_xgb.predict(X)[0])
            pred_lgbm = float(model_lgbm.predict(X)[0])
            pred_cat = float(model_cat.predict(X)[0])
            
            # Use simple ensemble average (can be tweaked to proper weights)
            ensemble_pred = (pred_xgb + pred_lgbm + pred_cat) / 3.0
            return float(ensemble_pred)
        except Exception as e:
            print("Error predicting:", e)

    if not history:
        return float(fallback if fallback is not None else 2200.0)

    latest = history[-1].actual
    rolling = float(np.mean([item.actual for item in history[-6:]]))
    lag24 = find_lag_value(history, timestamp, 24) or latest
    hour = timestamp.hour
    day = timestamp.weekday()
    daily_shape = math.sin(((hour - 5) / 24) * math.pi * 2) * 70
    evening_lift = math.exp(-((hour - 19) ** 2) / 18) * 120
    weekday_lift = 35 if day < 5 else -45

    return latest * 0.48 + rolling * 0.24 + lag24 * 0.18 + daily_shape + evening_lift + weekday_lift


def make_features(history: list[HourObservation], timestamp: datetime) -> list[float]:
    """Generates features compatible with the ML models based on history constraints."""
    import math
    values = [item.actual for item in history]
    
    def get_lag(h: int) -> float:
        val = find_lag_value(history, timestamp, h)
        return float(val) if val is not None else float(values[-1] if values else 0.0)

    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month
    is_weekend = int(day_of_week >= 5)
    quarter = (month - 1) // 3 + 1
    day_of_year = timestamp.timetuple().tm_yday

    lag_24h = get_lag(24)
    lag_48h = get_lag(48)
    lag_72h = get_lag(72)
    lag_168h = get_lag(168)
    
    r_24 = float(np.mean(values[-24:])) if len(values) >= 24 else float(values[-1] if values else 0.0)
    r_168 = float(np.mean(values[-168:])) if len(values) >= 168 else r_24
    r_std = float(np.std(values[-24:])) if len(values) >= 24 else 0.0
    
    prev_daily_mean = r_24 
    prev_daily_max = float(max(values[-24:])) if len(values) >= 24 else float(values[-1] if values else 0.0)
    ewm_24h = r_24

    # Trigonometric features
    sin_hour_1 = math.sin(2 * math.pi * hour / 24)
    cos_hour_1 = math.cos(2 * math.pi * hour / 24)
    sin_hour_2 = math.sin(4 * math.pi * hour / 24)
    cos_hour_2 = math.cos(4 * math.pi * hour / 24)
    sin_hour_3 = math.sin(6 * math.pi * hour / 24)
    cos_hour_3 = math.cos(6 * math.pi * hour / 24)

    sin_dow_1 = math.sin(2 * math.pi * day_of_week / 7)
    cos_dow_1 = math.cos(2 * math.pi * day_of_week / 7)
    sin_dow_2 = math.sin(4 * math.pi * day_of_week / 7)
    cos_dow_2 = math.cos(4 * math.pi * day_of_week / 7)
    sin_dow_3 = math.sin(6 * math.pi * day_of_week / 7)
    cos_dow_3 = math.cos(6 * math.pi * day_of_week / 7)

    return [
        float(hour), float(day_of_week), float(month), float(is_weekend), float(quarter), float(day_of_year),
        lag_24h, lag_48h, lag_72h, lag_168h,
        r_24, r_std, r_168, ewm_24h,
        sin_hour_1, cos_hour_1, sin_hour_2, cos_hour_2, sin_hour_3, cos_hour_3,
        sin_dow_1, cos_dow_1, sin_dow_2, cos_dow_2, sin_dow_3, cos_dow_3,
        prev_daily_mean, prev_daily_max
    ]


def find_lag_value(history: list[HourObservation], timestamp: datetime, lag_hours: int) -> float | None:
    target = timestamp - timedelta(hours=lag_hours)
    for item in history:
        if parse_hour(item.timestamp) == target:
            return item.actual
    return None


def prediction_interval(history: list[HourObservation], predicted: float) -> tuple[float, float]:
    values = [item.actual for item in history[-12:]]
    if len(values) < 2:
        band = max(70.0, predicted * 0.04)
    else:
        band = max(70.0, float(np.std(values)) * 0.8)
    return round(predicted - band, 2), round(predicted + band, 2)


def calculate_metrics(observed_points: list[dict]) -> ForecastMetrics:
    scored = [p for p in observed_points if p.get("error") is not None and p.get("actual") is not None]

    if not scored:
        return ForecastMetrics(mae=0, rmse=0, mse=0, mape=0)

    errors = np.asarray([abs(p["error"] or 0) for p in scored], dtype=float)
    squared = np.asarray([(p["error"] or 0) ** 2 for p in scored], dtype=float)
    mse_val = round(float(np.mean(squared)), 1)
    percentages = np.asarray(
        [abs(p["error"] or 0) / max(1.0, p["actual"]) * 100 for p in scored], dtype=float
    )

    return ForecastMetrics(
        mae=round(float(np.mean(errors)), 1),
        rmse=round(float(np.sqrt(mse_val)), 1),
        mse=mse_val,
        mape=round(float(np.mean(percentages)), 2),
    )


def build_recommendations(
    history: list[HourObservation],
    future: list[dict],
    metrics: ForecastMetrics,
) -> list[str]:
    if not history:
        return ["Add the first hourly actual value to start the online forecast buffer."]

    latest = history[-1]
    peak = max(future, key=lambda p: p["predicted"]) if future else None
    trough = min(future, key=lambda p: p["predicted"]) if future else None
    trend = "rising" if len(future) > 1 and future[-1]["predicted"] > future[0]["predicted"] else "easing"
    alert_count = sum(
        1 for p in [build_observed_point(history, i) for i in range(len(history))]
        if abs(p.get("error") or 0) / max(1.0, p.get("actual") or 1) * 100 >= 8
    )

    recs = [
        f"Latest reading {latest.actual:,.0f} MW at {latest.timestamp.replace('T', ' ')} — model is tracking closely.",
        f"Next {BUFFER_HOURS}h demand is {trend}; peak forecast {peak['predicted']:,.0f} MW at {peak['timestamp'].replace('T', ' ') if peak else 'N/A'}.",
    ]
    if alert_count > 0:
        recs.append(f"High deviation detected in {alert_count} recent hours. Confidence intervals expanded.")
    if trough and peak and (peak["predicted"] - trough["predicted"]) > 300:
        recs.append(f"Demand swing of {peak['predicted'] - trough['predicted']:,.0f} MW expected — ramp resources may be needed.")
    return recs


def build_insight(
    timestamp: datetime, actual: float | None, predicted: float, condition: str, is_forecast: bool
) -> str:
    if 17 <= timestamp.hour <= 22:
        block = "evening peak window"
    elif 6 <= timestamp.hour <= 11:
        block = "morning ramp"
    elif 0 <= timestamp.hour <= 5:
        block = "overnight low-demand period"
    else:
        block = "midday load period"

    ts_str = to_hour_value(timestamp).replace("T", " ")

    if is_forecast:
        return (
            f"Forecast {predicted:,.0f} MW at {ts_str} during the {block}. "
            f"Demand condition: {condition}."
        )

    error = round(actual - predicted, 1) if actual is not None else 0
    direction = "over-predicted" if error < 0 else "under-predicted"
    pct = abs(error) / max(1.0, actual) * 100 if actual else 0
    return (
        f"{ts_str}: actual {actual:,.0f} MW vs {predicted:,.0f} MW predicted "
        f"({'+' if error >= 0 else ''}{error:,.0f} MW, {pct:.1f}% {direction}). "
        f"Condition: {condition}."
    )


def condition_from_error(error_percent: float) -> str:
    if error_percent >= 10.0:
        return "High"
    elif error_percent >= 5.0:
        return "Watch"
    return "Normal"

def condition_from_forecast(predicted: float, history: list[HourObservation]) -> str:
    baseline = float(np.mean([item.actual for item in history[-12:]])) if history else predicted
    deviation = (predicted - baseline) / max(1.0, baseline) * 100
    if deviation > 20:
        return "Severe"
    if deviation > 8:
        return "High"
    if deviation < -10:
        return "Low"
    return "Normal"


def parse_hour(value: str) -> datetime:
    import pandas as pd
    dt = pd.to_datetime(value, dayfirst=False, format="mixed")
    return datetime(dt.year, dt.month, dt.day, dt.hour)


def to_hour_value(value: datetime) -> str:
    # Must remain ISO standard internally because frontend <input type="datetime-local"> requires YYYY-MM-DDTHH:MM
    return value.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
