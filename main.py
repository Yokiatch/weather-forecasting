import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your existing open-meteo fetching script
from scripts.fetch_current_weather import get_current_weather

# 1. Initialize the App
app = FastAPI(
    title="Weather Forecast API",
    description="Backend engine for the Uncertainty-Aware Weather Dashboard",
    version="1.0.0"
)

# 2. Configure CORS (Crucial for React frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load ML Artifacts on Startup
try:
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    base_uncertainty = joblib.load("models/uncertainty.pkl")
    print("✅ Machine Learning Models loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML models. Ensure they exist in the 'models/' directory. Error: {e}")

# 4. Define the Data Schema for Predictions
# This tells FastAPI exactly what incoming data to expect from React
class PredictionRequest(BaseModel):
    current_temp: float
    windspeed: float
    humidity: float
    cloudcover: float

# ---------------- API ENDPOINTS ----------------

@app.get("/api/current_weather")
def get_live_weather():
    """Fetches the latest live weather data from Open-Meteo."""
    try:
        weather_data = get_current_weather()
        return {"status": "success", "data": weather_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch live weather: {str(e)}")

@app.post("/api/predict_7_days")
def generate_forecast(request: PredictionRequest):
    """Generates the 7-day recursive forecast with uncertainty bounds."""
    try:
        temp_lags = [request.current_temp] * 7
        predictions = []

        for day in range(1, 8):
            # Format data exactly as the XGBoost model expects it
            data = {
                "humidity": request.humidity,
                "windspeed": request.windspeed,
                "cloudcover": request.cloudcover, 
            }

            for i in range(7):
                data[f"temp_lag{i+1}"] = temp_lags[i]

            df = pd.DataFrame([data])
            scaled = scaler.transform(df)
            pred = float(model.predict(scaled)[0])  # Ensure it's a standard Python float

            # Calculate Uncertainty
            temp_diff = abs(pred - request.current_temp)
            dynamic_uncertainty = base_uncertainty * (1 + temp_diff / 10)
            margin = 1.96 * dynamic_uncertainty

            predictions.append({
                "day": day,
                "temp": round(pred, 2),
                "low": round(pred - margin, 2),
                "high": round(pred + margin, 2),
                "uncertainty": round(margin, 2)
            })

            # Update the lags for the next recursive loop
            temp_lags = [pred] + temp_lags[:-1]

        return {"status": "success", "forecast": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction engine failed: {str(e)}")