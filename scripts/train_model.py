import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib, os

# Load data
df = pd.read_csv('data/weather.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Create lag features (last 7 days)
for lag in range(1, 8):
    df[f'temp_lag{lag}'] = df['temp'].shift(lag)

df = df.dropna()

X = df.drop(columns=['date', 'temp'])
y = df['temp']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgb_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model saved to models/")
