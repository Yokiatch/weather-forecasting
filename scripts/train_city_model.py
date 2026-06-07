import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def train_for_city(df: pd.DataFrame) -> tuple:
    """
    Given a historical weather DataFrame, train an XGBoost model.
    Returns (model, scaler, mae) tuple.
    """
    df = df.copy()

    # Generate 7 lag features
    for i in range(1, 8):
        df[f"temp_lag{i}"] = df["temp"].shift(i)
    df = df.dropna().reset_index(drop=True)

    FEATURES = ["humidity", "windspeed", "cloudcover"] + [f"temp_lag{i}" for i in range(1, 8)]
    X = df[FEATURES]
    y = df["temp"]

    # 80/20 split (time-ordered, no shuffle)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train_sc, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test_sc))
    return model, scaler, mae
