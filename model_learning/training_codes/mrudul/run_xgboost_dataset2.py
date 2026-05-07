import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

print("🔋 Loading data for Optuna Tuning & Multi-Horizon Evaluation...")
df = pd.read_csv("cleaned_energy_data_model2.csv")

# 1. Prepare Features & Drop Leakage
df['start_time'] = pd.to_datetime(df['start_time'])
drop_cols = ['start_time', 'end_time_utc', 'consumption', 'how_historical_mean', 'resid_vs_how_mean']
features = df.drop(columns=[c for c in drop_cols if c in df.columns])
target = df['consumption']

# 2. Split Data (Train: 2015-2019 | Val: 2020 | Test: 2021)
# Optuna ke liye validation set (2020) alag nikalna zaruri hai
train_mask = df['start_time'].dt.year <= 2019
val_mask = df['start_time'].dt.year == 2020
test_mask = df['start_time'].dt.year == 2021

X_train_opt, y_train_opt = features.loc[train_mask], target.loc[train_mask]
X_val_opt, y_val_opt = features.loc[val_mask], target.loc[val_mask]

# Final Training ke liye 2015-2020 combine karenge
X_train_full = features.loc[train_mask | val_mask]
y_train_full = target.loc[train_mask | val_mask]

X_test, y_test = features.loc[test_mask], target.loc[test_mask]
test_dates = df.loc[test_mask, 'start_time'].values

# =========================================================
# 3. OPTUNA HYPERPARAMETER TUNING
# =========================================================
print("🧠 Starting Optuna Hyperparameter Tuning (Finding the perfect model)...")

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_jobs': -1,
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**params)
    # Validate on 2020 data
    model.fit(X_train_opt, y_train_opt, eval_set=[(X_val_opt, y_val_opt)], verbose=False)
    preds = model.predict(X_val_opt)
    return mean_absolute_error(y_val_opt, preds)

# Create and run study (Sirf 15 trials rakhe hain hackathon speed ke liye, raat mein 50 kar lena)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("\n🏆 OPTUNA BEST PARAMETERS FOUND:")
print(study.best_params)

# =========================================================
# 4. TRAIN FINAL MODEL WITH BEST PARAMS
# =========================================================
print("\n🚀 Training Final Model with Best Parameters on full 2015-2020 data...")
best_params = study.best_params
best_params['n_jobs'] = -1
best_params['random_state'] = 42

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train_full, y_train_full, verbose=False)
print("✅ Final Training Complete!")

# =========================================================
# 5. PREDICTIONS (YEAR, MONTH, WEEK)
# =========================================================

# --- A. FULL YEAR (Direct Historical Backtest) ---
print("🔮 Forecasting Full Year 2021...")
preds_year = final_model.predict(X_test)
actual_year = y_test.values

# --- B. 1 MONTH (January 2021) ---
# Jan has 31 days * 24 hours = 744 hours
month_horizon = 744
preds_month = preds_year[:month_horizon]
actual_month = actual_year[:month_horizon]

# --- C. 1 WEEK (168 Hours - Strictly Autoregressive) ---
print("🔄 Running Strict Autoregressive Forecast for 1st Week...")
week_horizon = 168
recursive_predictions = []
history_actuals = list(y_train_full.tail(168).values)
X_test_168h = X_test.head(week_horizon).copy()

for i in range(week_horizon):
    current_row = X_test_168h.iloc[[i]].copy()
    if 'lag_1h' in current_row.columns: current_row['lag_1h'] = history_actuals[-1]
    if 'lag_24h' in current_row.columns: current_row['lag_24h'] = history_actuals[-24]
    if 'lag_48h' in current_row.columns: current_row['lag_48h'] = history_actuals[-48]
    if 'lag_72h' in current_row.columns: current_row['lag_72h'] = history_actuals[-72]
    if 'lag_168h' in current_row.columns: current_row['lag_168h'] = history_actuals[-168]
    if 'rolling_mean_24h' in current_row.columns: current_row['rolling_mean_24h'] = np.mean(history_actuals[-24:])
    
    pred = final_model.predict(current_row)[0]
    recursive_predictions.append(pred)
    history_actuals.append(pred)

actual_week = actual_year[:week_horizon]

# =========================================================
# 6. PRINTING THE EVALUATION METRICS
# =========================================================
def print_metrics(timeframe, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n📊 {timeframe.upper()} METRICS:")
    print(f"   MAE:  {mae:.2f} MWh")
    print(f"   RMSE: {rmse:.2f} MWh")
    print(f"   MAPE: {mape * 100:.2f}%  (Accuracy: {100 - (mape*100):.2f}%)")

print("\n" + "="*50)
print("🏆 FINAL MULTI-HORIZON EVALUATION")
print("="*50)
print_metrics("1-Week (Autoregressive)", actual_week, recursive_predictions)
print_metrics("1-Month (Historical Backtest)", actual_month, preds_month)
print_metrics("1-Year (Historical Backtest)", actual_year, preds_year)
print("="*50)

# =========================================================
# 7. EXPORT HOURLY PREDICTIONS TO CSV
# =========================================================
print("\n💾 Saving Hourly Energy Predictions...")
results_df = pd.DataFrame({
    'timestamp': test_dates,
    'actual_consumption_MWh': actual_year,
    'predicted_consumption_MWh': preds_year
})
results_df.to_csv("optuna_final_predictions_2021.csv", index=False)
print("✅ Saved all 8760 hourly predictions to 'optuna_final_predictions_2021.csv'")