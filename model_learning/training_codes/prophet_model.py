import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

print("🔋 Loading the 52k Row Dataset...")
# Make sure the CSV name is correct
df = pd.read_csv("cleaned_energy_data.csv")

# Prophet strictly needs 'ds' for dates and 'y' for values
df_prophet = df[['start_time', 'consumption']].copy()
df_prophet.rename(columns={'start_time': 'ds', 'consumption': 'y'}, inplace=True)
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet = df_prophet.dropna(subset=['y']) # Drop NaN just in case

# Split Data (Train: 2015-2020, Test: 2021)
print("✂️ Splitting Data (Train: 2015-2020 | Test: 2021)...")
train_df = df_prophet[df_prophet['ds'].dt.year <= 2020].copy()
test_df = df_prophet[df_prophet['ds'].dt.year == 2021].copy()

print(f"Training on {len(train_df)} rows...")
print(f"Testing on {len(test_df)} rows...")

# Initialize and Train
print("🚀 Training Facebook Prophet (Wait for it)...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True, # Hourly data needs this
)
model.fit(train_df)

# ── EVAL: Forecast on Training Data ──────────────────────────────────────────
print("📊 Evaluating on Training Data...")
train_future = train_df[['ds']].copy()
train_forecast = model.predict(train_future)
train_actuals = train_df['y'].values
train_predictions = train_forecast['yhat'].values

train_mae = mean_absolute_error(train_actuals, train_predictions)
train_mape = mean_absolute_percentage_error(train_actuals, train_predictions)
train_rmse = np.sqrt(mean_squared_error(train_actuals, train_predictions))

# Forecast on Test Data
print("🔮 Forecasting 2021...")
test_future = test_df[['ds']] 
forecast = model.predict(test_future)

# Calculate Test Accuracy
test_actuals = test_df['y'].values
test_predictions = forecast['yhat'].values

test_mae = mean_absolute_error(test_actuals, test_predictions)
test_mape = mean_absolute_percentage_error(test_actuals, test_predictions)
test_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))

print("\n" + "="*50)
print("📈 PROPHET COMPREHENSIVE EVALUATION RESULTS")
print("="*50)

print("\n🏋️ TRAINING METRICS:")
print(f"  Train MAE (Mean Absolute Error):   {train_mae:.2f} MWh")
print(f"  Train RMSE (Root Mean Sq Error):   {train_rmse:.2f} MWh")
print(f"  Train MAPE (Mean Abs % Error):     {train_mape * 100:.2f}%")
print(f"  Train Accuracy Score:              {100 - (train_mape * 100):.2f}%")

print("\n🧪 TEST METRICS:")
print(f"  Test MAE (Mean Absolute Error):    {test_mae:.2f} MWh")
print(f"  Test RMSE (Root Mean Sq Error):    {test_rmse:.2f} MWh")
print(f"  Test MAPE (Mean Abs % Error):      {test_mape * 100:.2f}%")
print(f"  Test Accuracy Score:               {100 - (test_mape * 100):.2f}%")

print("\n📊 OVERFITTING INDICATOR:")
overfit_ratio = test_mae / train_mae if train_mae > 0 else float('inf')
print(f"  Test MAE / Train MAE ratio:        {overfit_ratio:.2f}x")
if overfit_ratio > 1.5:
    print(f"  ⚠️  Model may be overfitting!")
else:
    print(f"  ✅ Model generalization looks good")

print("="*50)

# Save the new forecast for your frontend guy!
forecast[['ds', 'yhat']].to_csv("baseline_forecast_new.csv", index=False)
print("✅ Saved new forecast to 'baseline_forecast_new.csv'")