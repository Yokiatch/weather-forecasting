import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

print("1. Loading raw weather data...")
# Look for weather.csv in the root folder where the command is run
try:
    df_raw = pd.read_csv("weather.csv")
except FileNotFoundError:
    print("❌ Error: Could not find 'weather.csv'. Make sure it is in your project root folder!")
    exit()

print("2. Processing data and generating time lags...")
# Keep only what we need
cols_to_keep = ['datetime', 'temp', 'windspeed', 'humidity', 'cloudcover']
df_processed = df_raw[cols_to_keep].copy()

# Sort chronologically
df_processed['datetime'] = pd.to_datetime(df_processed['datetime'])
df_processed = df_processed.sort_values('datetime').reset_index(drop=True)

# Generate 7 days of memory (lags)
for i in range(1, 8):
    df_processed[f'temp_lag{i}'] = df_processed['temp'].shift(i)

# Drop the first 7 days since they lack historical memory
df_processed = df_processed.dropna().reset_index(drop=True)

print("3. Setting up the machine learning model...")
features = ["humidity", "windspeed", "cloudcover"] + [f"temp_lag{i}" for i in range(1, 8)]
X = df_processed[features]
y = df_processed["temp"]

# 80/20 Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Scale features so humidity (0-100) doesn't overpower windspeed (0-20)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("4. Training XGBoost model...")
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Calculate dynamic uncertainty baseline
predictions = model.predict(X_test_scaled)
base_uncertainty = mean_absolute_error(y_test, predictions)
print(f"   -> Model Mean Absolute Error: {base_uncertainty:.2f} °C")

print("5. Saving model artifacts...")
# Ensure the models folder exists
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(base_uncertainty, "models/uncertainty.pkl")

print("✅ Success! The new model is trained and saved.")