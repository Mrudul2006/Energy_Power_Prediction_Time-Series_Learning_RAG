"""
APPROACH 2 — RESIDUAL BOOSTING CHAIN
======================================
Layer 1 (Prophet)  : Captures global trend, weekly + yearly seasonality, holidays
Layer 2 (XGBoost)  : Learns Prophet's residual errors using tabular lag features
Layer 3 (LightGBM) : Learns XGBoost's remaining residual errors
Final              : Prophet_pred + XGB_correction + LGBM_correction

Why this works:
  Each model captures different error types:
  • Prophet   → long-range trend, macro seasonality, holiday spikes
  • XGBoost   → local autocorrelation, hour-of-day peaks, rolling patterns
  • LightGBM  → fine-grained residual non-linearity left by XGB

Folder structure assumed:
    models/
        __init__.py
        common.py
        boosted_trees.py
        prophet_model.py
    ensemble_residual_chain.py   ← this file
    cleaned_energy_data_model2.csv
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from models.prophet_model import ProphetForecaster, ProphetConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("\n" + "="*65)
print("⛓️  RESIDUAL BOOSTING CHAIN: PROPHET → XGB → LGBM")
print("="*65)

# =========================================================
# 1. DATA PREPARATION
# =========================================================
print("\n📦 1. Loading Data...")
df = pd.read_csv("cleaned_energy_data_model2.csv")
df['start_time'] = pd.to_datetime(df['start_time'])
df = df.sort_values('start_time').reset_index(drop=True)

# Time series (DatetimeIndex) for Prophet
series = df.set_index('start_time')['consumption'].astype(float)

# Tabular features for XGB/LGBM residual correction
drop_cols = ['start_time', 'end_time_utc', 'consumption', 'how_historical_mean', 'resid_vs_how_mean']
features = df.drop(columns=[c for c in drop_cols if c in df.columns])
target   = df['consumption'].astype(float)

train_mask = df['start_time'].dt.year <= 2019
val_mask   = df['start_time'].dt.year == 2020
test_mask  = df['start_time'].dt.year == 2021

y_series_train = series[series.index.year <= 2019]
y_series_full  = series[series.index.year <= 2020]

val_horizon  = val_mask.sum()
test_horizon = test_mask.sum()

X_train = features.loc[train_mask];  y_train = target.loc[train_mask]
X_val   = features.loc[val_mask];    y_val   = target.loc[val_mask]
X_full  = features.loc[train_mask | val_mask]; y_full = target.loc[train_mask | val_mask]
X_test  = features.loc[test_mask];   y_test  = target.loc[test_mask]

# =========================================================
# 2. LAYER 1 — PROPHET: GLOBAL TREND + SEASONALITY
# =========================================================
print("\n🔮 2. Layer 1 — Fitting Prophet on 2015-2019...")

prophet_cfg = ProphetConfig(
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    add_india_holidays=True,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
)

# Train on 2015-2019, forecast 2020 (for residual training)
prophet_base = ProphetForecaster(prophet_cfg)
prophet_base.fit(y_series_train)
prophet_val_result = prophet_base.forecast(horizon=val_horizon)
prophet_val_preds = prophet_val_result.mean

# Train on 2015-2020, forecast 2021 (for final test)
prophet_final = ProphetForecaster(prophet_cfg)
prophet_final.fit(y_series_full)
prophet_test_result = prophet_final.forecast(horizon=test_horizon)
prophet_test_preds  = prophet_test_result.mean

prophet_val_mae  = mean_absolute_error(y_val.values, prophet_val_preds)
prophet_test_mae = mean_absolute_error(y_test.values, prophet_test_preds)
print(f"   ✅ Prophet VAL  MAE : {prophet_val_mae:.2f}")
print(f"   📊 Prophet TEST MAE : {prophet_test_mae:.2f}")

# ─── Prophet Residuals ────────────────────────────────────
# These are the errors Prophet couldn't explain
residuals_train_for_xgb = y_train.values - prophet_base.forecast(horizon=len(y_train)).mean
residuals_val_for_xgb   = y_val.values   - prophet_val_preds

# For the final model (trained on 2015-2020), compute training residuals
prophet_full_train_preds = prophet_final.forecast(horizon=len(y_full)).mean
residuals_full_for_xgb   = y_full.values - prophet_full_train_preds

# =========================================================
# 3. LAYER 2 — XGBOOST: CORRECT PROPHET'S RESIDUALS
# =========================================================
print("\n🌳 3. Layer 2 — XGBoost correcting Prophet residuals (Optuna tuned)...")

def objective_xgb_residual(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 300, 1200),
        'max_depth'       : trial.suggest_int('max_depth', 4, 10),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_jobs': -1, 'random_state': 42,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, residuals_train_for_xgb, verbose=False)
    xgb_correction_val = model.predict(X_val)
    # Evaluate on total error (Prophet + XGB correction)
    total_val_preds = prophet_val_preds + xgb_correction_val
    return mean_absolute_error(y_val.values, total_val_preds)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb_residual, n_trials=30)
best_xgb_params = {**study_xgb.best_params, 'n_jobs': -1, 'random_state': 42}

# Train final XGB residual corrector on full 2015-2020 residuals
xgb_corrector = xgb.XGBRegressor(**best_xgb_params)
xgb_corrector.fit(X_full, residuals_full_for_xgb, verbose=False)

# Compute XGB corrections
xgb_correction_val  = xgb_corrector.predict(X_val)
xgb_correction_test = xgb_corrector.predict(X_test)

# Layer 2 output: Prophet + XGB correction
layer2_val  = prophet_val_preds  + xgb_correction_val
layer2_test = prophet_test_preds + xgb_correction_test

layer2_val_mae  = mean_absolute_error(y_val.values, layer2_val)
layer2_test_mae = mean_absolute_error(y_test.values, layer2_test)
print(f"   ✅ Prophet+XGB VAL  MAE : {layer2_val_mae:.2f}  (Δ {prophet_val_mae - layer2_val_mae:+.2f})")
print(f"   📊 Prophet+XGB TEST MAE : {layer2_test_mae:.2f}")

# ─── XGB Residuals (what XGB still couldn't fix) ─────────
residuals_val_for_lgbm  = y_val.values   - layer2_val
residuals_full_for_lgbm = y_full.values  - (prophet_full_train_preds + xgb_corrector.predict(X_full))

# =========================================================
# 4. LAYER 3 — LIGHTGBM: CORRECT XGB's REMAINING RESIDUALS
# =========================================================
print("\n🌿 4. Layer 3 — LightGBM correcting XGBoost's residuals (Optuna tuned)...")

def objective_lgbm_residual(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 300, 1200),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth'       : trial.suggest_int('max_depth', 5, 12),
        'num_leaves'      : trial.suggest_int('num_leaves', 20, 100),
        'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_jobs': -1, 'random_state': 42, 'verbose': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_val, residuals_val_for_lgbm)   # Train on val residuals of XGB
    lgbm_correction = model.predict(X_val)
    total = layer2_val + lgbm_correction
    return mean_absolute_error(y_val.values, total)

study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm_residual, n_trials=30)
best_lgbm_params = {**study_lgbm.best_params, 'n_jobs': -1, 'random_state': 42, 'verbose': -1}

# Final LGBM corrector: trained on full residuals after Prophet+XGB
lgbm_corrector = lgb.LGBMRegressor(**best_lgbm_params)
lgbm_corrector.fit(X_full, residuals_full_for_lgbm)

lgbm_correction_val  = lgbm_corrector.predict(X_val)
lgbm_correction_test = lgbm_corrector.predict(X_test)

# =========================================================
# 5. FINAL CHAIN: PROPHET + XGB + LGBM
# =========================================================
final_val_preds  = prophet_val_preds  + xgb_correction_val  + lgbm_correction_val
final_test_preds = prophet_test_preds + xgb_correction_test + lgbm_correction_test

# =========================================================
# 6. EVALUATION — LAYERED COMPARISON
# =========================================================
def print_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:<30} | MAE: {mae:<8.2f} | RMSE: {rmse:<8.2f} | MAPE: {mape*100:<5.2f}% | R²: {r2:.4f}")

actual_2021 = y_test.values

print("\n" + "="*80)
print("🏆 RESIDUAL CHAIN — LAYER-BY-LAYER IMPROVEMENT (2021 Test)")
print("="*80)
print_metrics("Layer 1: Prophet only",         actual_2021, prophet_test_preds)
print_metrics("Layer 2: Prophet + XGB",        actual_2021, layer2_test)
print_metrics("Layer 3: Prophet + XGB + LGBM", actual_2021, final_test_preds)
print("="*80)

# Per-hour accuracy breakdown
print("\n📊 Per-Hour MAPE (Layer 3 Final):")
test_hours   = df.loc[test_mask, 'start_time'].dt.hour.values
hour_mapes   = {}
for h in range(24):
    mask = test_hours == h
    mape = mean_absolute_percentage_error(actual_2021[mask], final_test_preds[mask])
    hour_mapes[h] = mape * 100

worst_hours  = sorted(hour_mapes, key=hour_mapes.get, reverse=True)[:3]
best_hours   = sorted(hour_mapes, key=hour_mapes.get)[:3]
print(f"   Best  hours: {best_hours}  → MAPE ~{min(hour_mapes.values()):.2f}%")
print(f"   Worst hours: {worst_hours} → MAPE ~{max(hour_mapes.values()):.2f}%")

# =========================================================
# 7. SAVE RESULTS & PLOT
# =========================================================
results_df = pd.DataFrame({
    'timestamp'          : df.loc[test_mask, 'start_time'].values,
    'actual'             : actual_2021,
    'prophet'            : prophet_test_preds,
    'prophet_xgb'        : layer2_test,
    'prophet_xgb_lgbm'   : final_test_preds,
    'xgb_correction'     : xgb_correction_test,
    'lgbm_correction'    : lgbm_correction_test,
})
results_df.to_csv("residual_chain_2021.csv", index=False)

# Plot: show residual shrinkage
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
plot_n = 336  # 2 weeks

test_timestamps = df.loc[test_mask, 'start_time'].values[:plot_n]
axes[0].plot(test_timestamps, actual_2021[:plot_n], label='Actual', color='black', lw=2)
axes[0].plot(test_timestamps, prophet_test_preds[:plot_n], label='Layer 1: Prophet', color='orange', lw=1.5, ls='--')
axes[0].plot(test_timestamps, layer2_test[:plot_n], label='Layer 2: +XGB', color='steelblue', lw=1.5, ls='--')
axes[0].plot(test_timestamps, final_test_preds[:plot_n], label='Layer 3: +LGBM', color='crimson', lw=2)
axes[0].set_title("Residual Chain — First 2 Weeks of 2021")
axes[0].legend(ncol=4); axes[0].grid(True, alpha=0.3)

# Residual evolution
resid1 = actual_2021[:plot_n] - prophet_test_preds[:plot_n]
resid2 = actual_2021[:plot_n] - layer2_test[:plot_n]
resid3 = actual_2021[:plot_n] - final_test_preds[:plot_n]
axes[1].plot(test_timestamps, resid1, label=f'Prophet residual (σ={np.std(resid1):.0f})', color='orange', alpha=0.7)
axes[1].plot(test_timestamps, resid2, label=f'After XGB (σ={np.std(resid2):.0f})',         color='steelblue', alpha=0.7)
axes[1].plot(test_timestamps, resid3, label=f'After LGBM (σ={np.std(resid3):.0f})',        color='crimson', alpha=0.9)
axes[1].axhline(0, color='black', lw=0.8, ls='-')
axes[1].set_title("Residuals Shrinking at Each Layer (lower = better)")
axes[1].legend(ncol=3); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("residual_chain_plot.png", dpi=150)
print("\n✅ Saved CSV  : residual_chain_2021.csv")
print("✅ Saved Plot : residual_chain_plot.png")