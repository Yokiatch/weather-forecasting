# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import requests

# Import your existing open-meteo fetching script
from scripts.fetch_current_weather import get_current_weather

app = FastAPI()

# 1. Configure CORS so React can talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Define the exact structure of data expected from React
class PredictionRequest(BaseModel):
    current_temp: float
    windspeed: float
    humidity: float
    cloudcover: float

model = None
scaler = None
uncertainty = None

@app.on_event("startup")
def load_model():
    global model, scaler, uncertainty
    try:
        model = joblib.load('models/xgb_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        uncertainty = joblib.load('models/uncertainty.pkl')
        print("✅ Model, scaler, and uncertainty loaded successfully")
    except Exception as e:
        model = None
        scaler = None
        uncertainty = None
        print(f"⚠️ Could not load model/scaler/uncertainty: {e}")

@app.get("/")
def home():
    return {
        "message": "Weather Forecast API",
        "model_loaded": model is not None
    }

# 3. Add back the Current Weather Endpoint
@app.get("/api/current_weather")
def get_live_weather():
    try:
        weather_data = get_current_weather()
        return {"status": "success", "data": weather_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch live weather: {str(e)}")

@app.post("/api/predict_7_days")
def generate_forecast(request: PredictionRequest):
    try:
        # 1. Fetch the upcoming 7-day environmental data from Open-Meteo
        om_url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=28.4744&longitude=77.5040"
            "&daily=windspeed_10m_max,precipitation_sum,cloudcover_mean"
            "&timezone=auto"
            "&forecast_days=8" # Add this line!
        )
        om_response = requests.get(om_url, timeout=10).json()
        om_daily = om_response["daily"]

        # 2. Run your XGBoost Temperature Prediction
        temp_lags = [request.current_temp] * 7
        predictions = []

        for day in range(1, 8):
            data = {
                "humidity": request.humidity,
                "windspeed": request.windspeed,
                "cloudcover": request.cloudcover, 
            }

            for i in range(7):
                data[f"temp_lag{i+1}"] = temp_lags[i]

            df = pd.DataFrame([data])
            scaled = scaler.transform(df)
            pred = float(model.predict(scaled)[0])

            temp_diff = abs(pred - request.current_temp)
            # 4. FIXED: Variable name matches the globally loaded 'uncertainty'
            dynamic_uncertainty = uncertainty * (1 + temp_diff / 10)
            margin = 1.96 * dynamic_uncertainty

            # 3. Merge the ML Temp with the Open-Meteo environmental data
            predictions.append({
                "day": day,
                "temp": round(pred, 1),
                "low": round(pred - margin, 1),
                "high": round(pred + margin, 1),
                "uncertainty": round(margin, 2),
                "wind": om_daily["windspeed_10m_max"][day],
                "precip": om_daily["precipitation_sum"][day],
                "clouds": om_daily["cloudcover_mean"][day]
            })

            temp_lags = [pred] + temp_lags[:-1]

        return {"status": "success", "forecast": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction engine failed: {str(e)}")