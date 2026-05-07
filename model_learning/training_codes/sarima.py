import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import optuna
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # SARIMA warnings bohot deta hai, isliye band kar rahe hain

print("🔋 Loading data for SARIMA Optuna Tuning...")
df = pd.read_csv("cleaned_energy_data_model2.csv")
df['start_time'] = pd.to_datetime(df['start_time'])
df = df.sort_values('start_time').set_index('start_time')

# 1. Prepare Data
all_train = df[df.index.year <= 2020]['consumption'].copy()
train_df_full = all_train.tail(8760)  # Last 1 year for FINAL training
test_df = df[df.index.year == 2021]['consumption'].copy()

# OPTUNA KE LIYE CHOTA SPLIT (To save your PC from melting)
# Train: Nov 2020 (720 hrs), Val: Dec 2020 (744 hrs)
val_df_opt = train_df_full.tail(744) 
train_df_opt = train_df_full.iloc[-1464:-744] 

print(f"Total Final Training Size: {len(train_df_full)} hours")
print(f"Test Size (2021): {len(test_df)} hours")

# =========================================================
# 2. OPTUNA HYPERPARAMETER TUNING (SARIMA)
# =========================================================
print("\n🧠 Starting Optuna Tuning for SARIMA (Using small dataset for speed)...")

def objective(trial):
    # SARIMA grid search parameters (Keeping space VERY tight for hackathon speed)
    p = trial.suggest_int('p', 0, 1)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 1)
    
    P = trial.suggest_int('P', 0, 1)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 1)
    
    try:
        model = SARIMAX(train_df_opt, 
                        order=(p, d, q), 
                        seasonal_order=(P, D, Q, 24), # 24 for daily seasonality
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        results = model.fit(disp=False)
        
        # Forecast for validation set
        forecast = results.get_forecast(steps=len(val_df_opt))
        preds = forecast.predicted_mean
        
        return mean_absolute_error(val_df_opt, preds)
    except Exception as e:
        # If parameters cause mathematical failure, return a huge error
        return float('inf')

# Creating study (Only 5 trials for testing, increase this to 15 if you have time!)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

print("\n🏆 OPTUNA BEST SARIMA PARAMETERS:")
best_params = study.best_params
print(best_params)

# =========================================================
# 3. TRAIN FINAL MODEL WITH BEST PARAMS
# =========================================================
print("\n🚀 Training Final SARIMA Model on FULL 2020 Data (This will take a few minutes)...")
final_model = SARIMAX(train_df_full, 
                      order=(best_params['p'], best_params['d'], best_params['q']), 
                      seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], 24),
                      enforce_stationarity=False,
                      enforce_invertibility=False)

final_results = final_model.fit(disp=False)
print("✅ Final Training Complete!")

# =========================================================
# 4. MULTI-HORIZON FORECASTING (Week, Month, Year)
# =========================================================
print("\n🔮 Forecasting 2021 (SARIMA handles recursion automatically)...")

# SARIMA directly predicts future steps. 8760 steps = 1 year.
full_forecast = final_results.get_forecast(steps=len(test_df))
preds_year = full_forecast.predicted_mean.values
actual_year = test_df.values

# --- Extracting Horizons ---
# 1 WEEK (168 hours)
preds_week = preds_year[:168]
actual_week = actual_year[:168]

# 1 MONTH (744 hours)
preds_month = preds_year[:744]
actual_month = actual_year[:744]

# =========================================================
# 5. PRINTING EVALUATION METRICS
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
print("🏆 FINAL SARIMA MULTI-HORIZON EVALUATION")
print("="*50)
print_metrics("1-Week (Short-Term)", actual_week, preds_week)
print_metrics("1-Month (Mid-Term)", actual_month, preds_month)
print_metrics("1-Year (Long-Term)", actual_year, preds_year)
print("="*50)

# =========================================================
# 6. EXPORT PREDICTIONS
# =========================================================
print("\n💾 Saving Predictions...")
pred_df = pd.DataFrame({
    'timestamp': test_df.index, 
    'actual_MWh': actual_year, 
    'predicted_MWh': preds_year
})
pred_df.to_csv("sarima_optuna_forecast_2021.csv", index=False)
print("✅ Saved all predictions to 'sarima_optuna_forecast_2021.csv'")
