import pandas as pd
# Assuming the script you pasted is saved as 'prophet_model.py'
from models.prophet_model import ProphetForecaster, ProphetConfig 

print("Loading data...")
# Load your dataset
df = pd.read_excel("Energy Consumption Dataset.xlsx")

# Initialize the config and the model
print("Initializing Prophet...")
config = ProphetConfig(
    yearly_seasonality=True, 
    weekly_seasonality=True, 
    daily_seasonality=True
)
model = ProphetForecaster(config=config)

# Train it! (We pass the whole dataframe and tell it which columns to use)
print("Training Baseline Model (This might take a minute)...")
model.fit(
    data=df,
    target_col="Electricity consumption (MWh)",
    timestamp_col="Start time UTC"
)
print("Training Complete!")

# Predict the next 24 hours (horizon = 24)
print("Forecasting next 24 hours...")
forecast = model.forecast(horizon=24)

# Look at the results
print(f"Predicted mean values: {forecast.mean[:5]}")