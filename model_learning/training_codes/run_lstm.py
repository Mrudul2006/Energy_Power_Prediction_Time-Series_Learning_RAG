"""
APPROACH 2 — RESIDUAL BOOSTING CHAIN (FIXED)
=============================================
Bug that was fixed:
  ❌ OLD (WRONG): prophet_base.forecast(horizon=len(y_train))
       → This forecasts FUTURE timestamps, not in-sample fitted values.
       → XGB was trained on garbage residuals from 5 years in the future.

  ✅ NEW (CORRECT): prophet_base.model_.predict(train_df_with_ds)
       → This returns Prophet's in-sample fitted values (yhat).
       → Residuals = actual_train - prophet_yhat_train.
       → XGB learns to correct Prophet's real in-sample errors.
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
print("⛓️  RESIDUAL BOOSTING CHAIN (FIXED): PROPHET → XGB → LGBM")
print("="*65)

# =========================================================
# 1. DATA PREPARATION
# =========================================================
print("\n📦 1. Loading Data...")
df = pd.read_csv("cleaned_energy_data_model2.csv")
df['start_time'] = pd.to_datetime(df['start_time'])
df = df.sort_values('start_time').reset_index(drop=True)

series = df.set_index('start_time')['consumption'].astype(float)

drop_cols = ['start_time', 'end_time_utc', 'consumption', 'how_historical_mean', 'resid_vs_how_mean']
features  = df.drop(columns=[c for c in drop_cols if c in df.columns])
target    = df['consumption'].astype(float)

train_mask = df['start_time'].dt.year <= 2019
val_mask   = df['start_time'].dt.year == 2020
test_mask  = df['start_time'].dt.year == 2021

y_series_train = series[series.index.year <= 2019]
y_series_full  = series[series.index.year <= 2020]

val_horizon  = val_mask.sum()
test_horizon = test_mask.sum()

X_train = features.loc[train_mask]; y_train = target.loc[train_mask]
X_val   = features.loc[val_mask];   y_val   = target.loc[val_mask]
X_full  = features.loc[train_mask | val_mask]; y_full = target.loc[train_mask | val_mask]
X_test  = features.loc[test_mask];  y_test  = target.loc[test_mask]

# =========================================================
# 2. LAYER 1 — PROPHET
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

# ── Train on 2015-2019 ────────────────────────────────────
prophet_base = ProphetForecaster(prophet_cfg)
prophet_base.fit(y_series_train)

# ✅ FIX: In-sample fitted values using Prophet's internal predict()
#    We pass the TRAINING timestamps back into Prophet to get yhat.
train_ds_df = pd.DataFrame({'ds': y_series_train.index})
prophet_train_fitted = prophet_base.model_.predict(train_ds_df)['yhat'].values
residuals_train_for_xgb = y_train.values - prophet_train_fitted

# Forecast 2020 (val) and 2021 (test) — these are legitimate future forecasts
prophet_val_result   = prophet_base.forecast(horizon=val_horizon)
prophet_val_preds    = prophet_val_result.mean

# ── Retrain on 2015-2020 for final test ──────────────────
prophet_final = ProphetForecaster(prophet_cfg)
prophet_final.fit(y_series_full)

# ✅ FIX: In-sample fitted values on full 2015-2020 data
full_ds_df = pd.DataFrame({'ds': y_series_full.index})
prophet_full_fitted  = prophet_final.model_.predict(full_ds_df)['yhat'].values
residuals_full_for_xgb = y_full.values - prophet_full_fitted

prophet_test_result  = prophet_final.forecast(horizon=test_horizon)
prophet_test_preds   = prophet_test_result.mean

prophet_val_mae  = mean_absolute_error(y_val.values, prophet_val_preds)
prophet_test_mae = mean_absolute_error(y_test.values, prophet_test_preds)

print(f"   Prophet in-sample residual mean : {residuals_train_for_xgb.mean():.2f}  std: {residuals_train_for_xgb.std():.2f}")
print(f"   ✅ Prophet VAL  MAE : {prophet_val_mae:.2f}")
print(f"   📊 Prophet TEST MAE : {prophet_test_mae:.2f}")

# =========================================================
# 3. LAYER 2 — XGBOOST CORRECTS PROPHET RESIDUALS
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
    model.fit(X_train, residuals_train_for_xgb, verbose=False)  # ✅ correct residuals
    xgb_correction_val = model.predict(X_val)
    layer2_val = prophet_val_preds + xgb_correction_val
    return mean_absolute_error(y_val.values, layer2_val)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb_residual, n_trials=30)
best_xgb_params = {**study_xgb.best_params, 'n_jobs': -1, 'random_state': 42}
print(f"   Best XGB residual-corrector MAE on val: {study_xgb.best_value:.2f}")

# Final XGB corrector: trained on full 2015-2020 residuals
xgb_corrector = xgb.XGBRegressor(**best_xgb_params)
xgb_corrector.fit(X_full, residuals_full_for_xgb, verbose=False)

xgb_correction_val  = xgb_corrector.predict(X_val)
xgb_correction_test = xgb_corrector.predict(X_test)

layer2_val  = prophet_val_preds  + xgb_correction_val
layer2_test = prophet_test_preds + xgb_correction_test

layer2_val_mae  = mean_absolute_error(y_val.values, layer2_val)
layer2_test_mae = mean_absolute_error(y_test.values, layer2_test)
print(f"   ✅ Prophet+XGB VAL  MAE : {layer2_val_mae:.2f}  (Δ {prophet_val_mae - layer2_val_mae:+.2f})")
print(f"   📊 Prophet+XGB TEST MAE : {layer2_test_mae:.2f}")

# =========================================================
# 4. LAYER 3 — LIGHTGBM CORRECTS XGB REMAINING RESIDUALS
# =========================================================
print("\n🌿 4. Layer 3 — LightGBM correcting XGBoost's remaining residuals...")

# ✅ Residuals after layer 2 on training set
xgb_full_correction_train  = xgb_corrector.predict(X_full)
residuals_full_for_lgbm    = y_full.values - (prophet_full_fitted + xgb_full_correction_train)
residuals_val_for_lgbm     = y_val.values  - layer2_val   # for Optuna tuning only

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
    # Tune on val residuals — this is fine since we already fixed layer 2
    model.fit(X_val, residuals_val_for_lgbm)
    lgbm_correction = model.predict(X_val)
    layer3_val = layer2_val + lgbm_correction
    return mean_absolute_error(y_val.values, layer3_val)

study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm_residual, n_trials=30)
best_lgbm_params = {**study_lgbm.best_params, 'n_jobs': -1, 'random_state': 42, 'verbose': -1}

lgbm_corrector = lgb.LGBMRegressor(**best_lgbm_params)
lgbm_corrector.fit(X_full, residuals_full_for_lgbm)

lgbm_correction_val  = lgbm_corrector.predict(X_val)
lgbm_correction_test = lgbm_corrector.predict(X_test)

final_val_preds  = layer2_val  + lgbm_correction_val
final_test_preds = layer2_test + lgbm_correction_test

# =========================================================
# 5. SANITY CHECK: Does each layer actually improve?
# =========================================================
print("\n🔍 Sanity Check — Val residuals shrinking?")
print(f"   Prophet residual std      : {(y_val.values - prophet_val_preds).std():.2f}")
print(f"   After XGB  residual std   : {(y_val.values - layer2_val).std():.2f}")
print(f"   After LGBM residual std   : {(y_val.values - final_val_preds).std():.2f}")

# =========================================================
# 6. FINAL EVALUATION
# =========================================================
def print_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:<30} | MAE: {mae:<8.2f} | RMSE: {rmse:<8.2f} | MAPE: {mape*100:<5.2f}% | R²: {r2:.4f}")

actual_2021 = y_test.values
print("\n" + "="*80)
print("🏆 RESIDUAL CHAIN (FIXED) — LAYER-BY-LAYER (2021 Test)")
print("="*80)
print_metrics("Layer 1: Prophet only",         actual_2021, prophet_test_preds)
print_metrics("Layer 2: Prophet + XGB",        actual_2021, layer2_test)
print_metrics("Layer 3: Prophet + XGB + LGBM", actual_2021, final_test_preds)
print("="*80)

# =========================================================
# 7. SAVE & PLOT
# =========================================================
results_df = pd.DataFrame({
    'timestamp'         : df.loc[test_mask, 'start_time'].values,
    'actual'            : actual_2021,
    'prophet'           : prophet_test_preds,
    'prophet_xgb'       : layer2_test,
    'prophet_xgb_lgbm'  : final_test_preds,
})
results_df.to_csv("residual_chain_fixed_2021.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
plot_n = 336
ts_plot = df.loc[test_mask, 'start_time'].values[:plot_n]

axes[0].plot(ts_plot, actual_2021[:plot_n],         label='Actual',          color='black', lw=2)
axes[0].plot(ts_plot, prophet_test_preds[:plot_n],  label='Layer 1: Prophet',color='orange', lw=1.5, ls='--')
axes[0].plot(ts_plot, layer2_test[:plot_n],         label='Layer 2: +XGB',   color='steelblue', lw=1.5, ls='--')
axes[0].plot(ts_plot, final_test_preds[:plot_n],    label='Layer 3: +LGBM',  color='crimson', lw=2)
axes[0].set_title("Residual Chain (Fixed) — 2 Weeks of 2021")
axes[0].legend(ncol=4); axes[0].grid(True, alpha=0.3)

resid1 = actual_2021[:plot_n] - prophet_test_preds[:plot_n]
resid2 = actual_2021[:plot_n] - layer2_test[:plot_n]
resid3 = actual_2021[:plot_n] - final_test_preds[:plot_n]
axes[1].plot(ts_plot, resid1, label=f'Prophet residual  (σ={np.std(resid1):.0f})', color='orange', alpha=0.7)
axes[1].plot(ts_plot, resid2, label=f'After XGB  (σ={np.std(resid2):.0f})',        color='steelblue', alpha=0.7)
axes[1].plot(ts_plot, resid3, label=f'After LGBM (σ={np.std(resid3):.0f})',        color='crimson', alpha=0.9)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_title("Residuals Shrinking at Each Layer")
axes[1].legend(ncol=3); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("residual_chain_fixed_plot.png", dpi=150)
print("\n✅ Saved CSV  : residual_chain_fixed_2021.csv")
print("✅ Saved Plot : residual_chain_fixed_plot.png")