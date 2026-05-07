import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_FILE = "cleaned_energy_data_model.csv"
TIMESTAMP_COLUMN = "start_time"
TARGET_COLUMN = "consumption"
FORECAST_HOURS = 168

print("Loading data...")
df = pd.read_csv(CSV_FILE)
df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
df.set_index(TIMESTAMP_COLUMN, inplace=True)
df = df.sort_index()

if "end_time_utc" in df.columns:
    df.drop(columns=["end_time_utc"], inplace=True)

if "target" not in df.columns:
    df["target"] = df[TARGET_COLUMN].shift(-1)
df.dropna(inplace=True)

train_df = df[df.index < "2021-01-01"]
test_df = df[df.index >= "2021-01-01"]

if len(test_df) == 0:
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

FEATURES = [col for col in train_df.columns if col not in [TARGET_COLUMN, "target"]]

import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Use the last 20% of training data as a temporal validation set
    val_idx = int(len(train_df) * 0.8)
    X_tr, y_tr = train_df[FEATURES].iloc[:val_idx], train_df["target"].iloc[:val_idx]
    X_va, y_va = train_df[FEATURES].iloc[val_idx:], train_df["target"].iloc[val_idx:]
    
    _model = XGBRegressor(**params)
    _model.fit(X_tr, y_tr)
    preds = _model.predict(X_va)
    return mean_absolute_error(y_va, preds)

print("Starting Optuna Hyperparameter Tuning...")
# We use a small number of trials (e.g. 10) here for brevity; increase this for real usage
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

print(f"Best Trial: {study.best_trial.value}")
print("Best Parameters:", study.best_params)

print("Training Final XGBoost Model...")
best_params = study.best_params
best_params['random_state'] = 42
best_params['n_jobs'] = -1

model = XGBRegressor(**best_params)
model.fit(train_df[FEATURES], train_df["target"])

train_preds = model.predict(train_df[FEATURES])
std_error = np.std(train_df["target"] - train_preds)

print("Evaluating Predictions over complete test data for 168-hour horizons...")

# We will run a recursive forecast starting from each point in the test set.
# Operating on every single hour exhaustively as requested.
eval_step = 1  # Evaluate one 7-day forecast per hour in the test set

horizons = range(1, FORECAST_HOURS + 1)
errors_per_horizon = {h: [] for h in horizons}

test_indices = range(0, len(test_df) - FORECAST_HOURS, eval_step)

# We use the full df to fetch historical lags if needed.
df_features = test_df[FEATURES].copy()

# Pre-calculate to avoid DataFrame overhead in the loop
X_test_arr = df_features.to_numpy()
y_test_arr = test_df["target"].to_numpy()

from tqdm import tqdm
for start_idx in tqdm(test_indices, desc="Evaluating 168h horizons"):
    # We find the absolute index in the original dataframe
    abs_start_idx = df.index.get_loc(test_df.index[start_idx])
    
    # Store predictions for this window
    window_preds = []
    
    # We maintain a local history series of the target, starting with real historical data
    # Needs at least 200 points back to calculate 168 lags and ewm
    hist_series = df[TARGET_COLUMN].iloc[abs_start_idx - 250 : abs_start_idx].copy()
    
    # The static time features for the 168-hour window
    current_features = test_df[FEATURES].iloc[start_idx : start_idx + FORECAST_HOURS].copy()
    
    for h in range(FORECAST_HOURS):
        # Update our history with the previous predicted value (true recursion)
        if h > 0:
            hist_series.loc[current_features.index[h-1]] = window_preds[-1]
            
        # Dynamically recalculate lag and rolling features using our generated predictions:
        current_time = current_features.index[h]
        
        if 'lag_1h' in FEATURES: current_features.at[current_time, 'lag_1h'] = hist_series.iloc[-1]
        if 'lag_24h' in FEATURES: current_features.at[current_time, 'lag_24h'] = hist_series.iloc[-24]
        if 'lag_48h' in FEATURES: current_features.at[current_time, 'lag_48h'] = hist_series.iloc[-48]
        if 'lag_72h' in FEATURES: current_features.at[current_time, 'lag_72h'] = hist_series.iloc[-72]
        if 'lag_168h' in FEATURES: current_features.at[current_time, 'lag_168h'] = hist_series.iloc[-168]
        
        if 'rolling_mean_24h' in FEATURES: current_features.at[current_time, 'rolling_mean_24h'] = hist_series.iloc[-24:].mean()
        if 'rolling_std_24h' in FEATURES: current_features.at[current_time, 'rolling_std_24h'] = hist_series.iloc[-24:].std()
        if 'rolling_mean_168h' in FEATURES: current_features.at[current_time, 'rolling_mean_168h'] = hist_series.iloc[-168:].mean()
        if 'ewm_24h' in FEATURES: current_features.at[current_time, 'ewm_24h'] = hist_series.ewm(span=24).mean().iloc[-1]
            
        # Predict the next hour based entirely on recursively populated features
        curr_x = current_features.iloc[[h]][FEATURES]
        pred = model.predict(curr_x)[0]
        window_preds.append(pred)
        
    # Calculate errors against actuals
    actuals = y_test_arr[start_idx : start_idx + FORECAST_HOURS]
    for h_idx, (p, a) in enumerate(zip(window_preds, actuals)):
        errors_per_horizon[h_idx + 1].append(np.abs(p - a))

# Calculate mean error per horizon
mean_errors = {h: np.mean(errs) for h, errs in errors_per_horizon.items() if len(errs) > 0}

print("\n================ METRICS PER HORIZON (Over Complete Test Set) ================")
print("Horizon (Hours) | Mean Absolute Error (MAE)")
print("-------------------------------------------")
for h in range(1, FORECAST_HOURS + 1):
    if h in mean_errors:
        print(f"{h:15d} | {mean_errors[h]:.4f}")

overall_mae = np.mean([np.mean(errs) for errs in errors_per_horizon.values() if len(errs) > 0])
print(f"\nOverall 7-Day Recursive MAE : {overall_mae:.4f}")
