# scripts/predict.py
import pandas as pd
import joblib
import sys

try:
    model = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    sys.exit(1)

latest_data = pd.DataFrame([{
    'humidity': 70,
    'windspeed': 8.0,
    'cloudcover': 50.0,
    'temp_lag1': 24.5,
    'temp_lag2': 25.1,
    'temp_lag3': 24.8,
    'temp_lag4': 25.0,
    'temp_lag5': 24.7,
    'temp_lag6': 25.2,
    'temp_lag7': 24.9
}])

scaled = scaler.transform(latest_data)
pred = model.predict(scaled)
print(f"Predicted temperature: {pred[0]:.2f}°C")
