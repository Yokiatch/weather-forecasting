import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('weather.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Recreate lag features exactly as train_model.py does
for lag in range(1, 8):
    df[f'temp_lag{lag}'] = df['temp'].shift(lag)
df = df.dropna().reset_index(drop=True)

# Use the exact same feature set as training
FEATURES = ["humidity", "windspeed", "cloudcover"] + [f"temp_lag{i}" for i in range(1, 8)]
X = df[FEATURES]
y = df['temp']

scaler = joblib.load('models/scaler.pkl')
model  = joblib.load('models/xgb_model.pkl')

X_scaled = scaler.transform(X)
pred = model.predict(X_scaled)

mae  = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

os.makedirs("results", exist_ok=True)

# Save actual vs predicted CSV for generate_graphs.py
out_df = df[['datetime', 'temp']].copy()
out_df['pred'] = pred
out_df.to_csv("results/actual_vs_pred.csv", index=False)

with open("results/metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")

plt.figure(figsize=(10, 5))
plt.plot(df['datetime'], y,    label='Actual Temperature',  linewidth=2)
plt.plot(df['datetime'], pred, label='Predicted Temperature', linewidth=2, linestyle='--')
plt.legend()
plt.title("Actual vs Predicted Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig("results/performance_plot.png")
print("Saved plot to results/performance_plot.png")
