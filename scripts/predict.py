import pandas as pd
import joblib
import numpy as np

model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Example: latest 7 days data (replace with real)
latest_data = pd.DataFrame({
    'humidity': [70, 68, 72, 71, 69, 73, 70],
    'pressure': [1012, 1011, 1013, 1010, 1011, 1014, 1013],
    'windspeed': [8.0, 7.9, 8.1, 8.3, 7.8, 8.2, 8.0],
    'temp_lag1': [24.5], 'temp_lag2': [25.1], 'temp_lag3': [24.8],
    'temp_lag4': [25.0], 'temp_lag5': [24.7], 'temp_lag6': [25.2], 'temp_lag7': [24.9]
})

scaled = scaler.transform(latest_data)
pred = model.predict(scaled)
print(f"Predicted temperature: {pred[0]:.2f}°C")
