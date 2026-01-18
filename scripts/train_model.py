# scripts/train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

np.random.seed(42)

DATA_PATH = "data/weather.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/fetch_weather_data.py first.")

df = pd.read_csv(DATA_PATH)
if df.shape[0] == 0:
    raise ValueError(f"{DATA_PATH} is empty. Provide valid data.")

# Ensure date column
if 'date' not in df.columns:
    df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
else:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isna().any():
        raise ValueError("Some 'date' values could not be parsed. Fix formats in data/weather.csv")

# Ensure required columns exist
for col in ['temp', 'humidity', 'pressure', 'windspeed']:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' is missing from {DATA_PATH}")

df = df.sort_values('date').reset_index(drop=True)

# Create lag features (last 7 days)
for lag in range(1, 8):
    df[f'temp_lag{lag}'] = df['temp'].shift(lag)

df = df.dropna().reset_index(drop=True)
if df.shape[0] < 10:
    raise ValueError("Not enough rows after creating lag features. Need at least 10 rows.")

X = df.drop(columns=['date', 'temp'])
y = df['temp']

# Split train/test (time-series split: no shuffle)
train_size = int(0.8 * len(X))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, pred)

# FIXED RMSE (compatible with all sklearn versions)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)

# Calculate prediction uncertainty (standard deviation of errors)
residuals = y_test - pred
uncertainty = np.std(residuals)
print(f"Prediction uncertainty (std dev): {uncertainty:.4f}")


print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Save model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgb_model.pkl', compress=3)
joblib.dump(scaler, 'models/scaler.pkl', compress=3)
joblib.dump(uncertainty, 'models/uncertainty.pkl')
print("Model saved to models/")

# Save metrics and sample predictions
os.makedirs('results', exist_ok=True)
with open('results/metrics.txt', 'w') as f:
    f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")

df_eval = df.iloc[train_size:].copy()
df_eval['pred'] = pred
df_eval.to_csv('results/actual_vs_pred.csv', index=False)
print("Saved results/actual_vs_pred.csv and results/metrics.txt")
