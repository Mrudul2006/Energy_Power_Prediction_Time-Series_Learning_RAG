import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

print("🔋 Loading data for XGBoost...")
df = pd.read_csv("cleaned_energy_data.csv")

# 1. Prepare Features
# We drop time columns for training, but keep them for splitting
df['start_time'] = pd.to_datetime(df['start_time'])
# Drop time columns AND leakage features (global statistics calculated on entire dataset)
features = df.drop(columns=['start_time', 'end_time_utc', 'consumption', 
                             'consumption_log', 'consumption_sqrt', 'consumption_smooth',
                             'same_hour_global_avg', 'hourly_avg_deviation', 
                             'dow_avg_deviation', 'month_avg_deviation'])
target = df['consumption']

# 2. The 2021 OOT Split Strategy
print("✂️ Splitting Data (Train: 2015-2020 | Test: 2021)...")
train_idx = df[df['start_time'].dt.year <= 2020].index
test_idx = df[df['start_time'].dt.year == 2021].index

X_train, X_test = features.loc[train_idx], features.loc[test_idx]
y_train, y_test = target.loc[train_idx], target.loc[test_idx]

print(f"Training on {len(X_train)} rows...")
print(f"Testing on {len(X_test)} rows...")

# 3. Initialize XGBoost
# Using standard hackathon parameters for time-series
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1, # Use ALL your CPU cores
    early_stopping_rounds=50,
    random_state=42
)

# 4. Train the Model
print("🚀 Training XGBoost (This will be FAST)...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# 5. Predictions on Training Data
print("📊 Evaluating on Training Data...")
train_preds = model.predict(X_train)

train_mae = mean_absolute_error(y_train, train_preds)
train_mape = mean_absolute_percentage_error(y_train, train_preds)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))

# 6. Predictions on Test Data
print("🔮 Forecasting 2021...")
test_preds = model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_preds)
test_mape = mean_absolute_percentage_error(y_test, test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

print("\n" + "="*50)
print("🌳 XGBOOST COMPREHENSIVE EVALUATION RESULTS")
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

# 7. Save Results for Frontend
results_df = pd.DataFrame({'timestamp': df.loc[test_idx, 'start_time'], 'actual': y_test, 'predicted': test_preds})
results_df.to_csv("xgboost_forecast.csv", index=False)

# 8. Bonus: Feature Importance Plot
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=15)
plt.title("Top 15 Drivers of Energy Consumption")
plt.savefig("feature_importance.png")
print("✅ Saved forecast to 'xgboost_forecast.csv'")
print("✅ Saved feature importance plot to 'feature_importance.png'")