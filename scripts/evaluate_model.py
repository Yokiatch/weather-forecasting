import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('data/weather.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

for lag in range(1, 8):
    df[f'temp_lag{lag}'] = df['temp'].shift(lag)
df = df.dropna()

X = df.drop(columns=['date', 'temp'])
y = df['temp']

scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/xgb_model.pkl')

X_scaled = scaler.transform(X)
pred = model.predict(X_scaled)

mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

plt.figure(figsize=(10,5))
plt.plot(df['date'], y, label='Actual Temperature', linewidth=2)
plt.plot(df['date'], pred, label='Predicted Temperature', linewidth=2)
plt.legend()
plt.title("Actual vs Predicted Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig("results/performance_plot.png")
print("Saved plot to results/performance_plot.png")
