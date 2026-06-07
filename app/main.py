# app/main.py
import time
import asyncio
import requests
import joblib
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scripts.fetch_current_weather import get_current_weather
from scripts.fetch_historical import fetch_historical_weather, fetch_recent_7days
from scripts.train_city_model import train_for_city

app = FastAPI(
    title="Weather Forecast API",
    description="Global Uncertainty-Aware 7-Day Weather Forecast",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache: key = "lat_lon", value = {model, scaler, mae, trained_at} ──
_model_cache: dict = {}
CACHE_TTL_HOURS = 24  # retrain after 24 hours

# ── Pre-loaded Greater Noida model (from disk) ─────────────────────────────────
_default_model   = None
_default_scaler  = None
_default_mae     = None
DEFAULT_LAT = 28.4744
DEFAULT_LON = 77.5040

@app.on_event("startup")
def load_default_model():
    global _default_model, _default_scaler, _default_mae
    try:
        _default_model  = joblib.load("models/xgb_model.pkl")
        _default_scaler = joblib.load("models/scaler.pkl")
        _default_mae    = joblib.load("models/uncertainty.pkl")
        print("✅ Default Greater Noida model loaded")
    except Exception as e:
        print(f"⚠️  Could not load default model: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, 3)}_{round(lon, 3)}"

def _is_default_location(lat: float, lon: float) -> bool:
    return abs(lat - DEFAULT_LAT) < 0.05 and abs(lon - DEFAULT_LON) < 0.05

def _get_cached(lat: float, lon: float) -> Optional[dict]:
    key   = _cache_key(lat, lon)
    entry = _model_cache.get(key)
    if not entry:
        return None
    age_hours = (datetime.now() - entry["trained_at"]).total_seconds() / 3600
    if age_hours > CACHE_TTL_HOURS:
        del _model_cache[key]
        return None
    return entry

def _set_cache(lat: float, lon: float, model, scaler, mae: float):
    _model_cache[_cache_key(lat, lon)] = {
        "model":      model,
        "scaler":     scaler,
        "mae":        mae,
        "trained_at": datetime.now(),
    }

def _run_forecast(model, scaler, mae, current_temp, windspeed, humidity,
                  cloudcover, lat, lon) -> list:
    """Core recursive XGBoost forecast + Open-Meteo env data merge."""
    # Fetch real last-7-days temps for lag initialization
    try:
        temp_lags = fetch_recent_7days(lat, lon)
        # If we got fewer than 7, pad with current temp
        while len(temp_lags) < 7:
            temp_lags.append(current_temp)
    except Exception:
        temp_lags = [current_temp] * 7

    # Fetch 8-day environmental forecast from Open-Meteo
    om_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=windspeed_10m_max,precipitation_sum,cloudcover_mean"
        f"&timezone=auto&forecast_days=8"
    )
    om_resp  = requests.get(om_url, timeout=10)
    om_resp.raise_for_status()
    om_daily = om_resp.json()["daily"]

    predictions = []
    for day in range(1, 8):
        features = {
            "humidity":   humidity,
            "windspeed":  windspeed,
            "cloudcover": cloudcover,
        }
        for i in range(7):
            features[f"temp_lag{i+1}"] = temp_lags[i]

        df     = pd.DataFrame([features])
        scaled = scaler.transform(df)
        pred   = float(model.predict(scaled)[0])

        temp_diff           = abs(pred - current_temp)
        dynamic_uncertainty = mae * (1 + temp_diff / 10)
        margin              = 1.96 * dynamic_uncertainty

        predictions.append({
            "day":         day,
            "temp":        round(pred, 1),
            "low":         round(pred - margin, 1),
            "high":        round(pred + margin, 1),
            "uncertainty": round(margin, 2),
            "wind":        om_daily["windspeed_10m_max"][day],
            "precip":      om_daily["precipitation_sum"][day],
            "clouds":      om_daily["cloudcover_mean"][day],
        })

        temp_lags = [pred] + temp_lags[:-1]

    return predictions


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    current_temp: float
    windspeed:    float
    humidity:     float
    cloudcover:   float
    lat:          float = DEFAULT_LAT
    lon:          float = DEFAULT_LON


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    cached_cities = len(_model_cache)
    return {
        "message":       "Weather Forecast API v3",
        "default_model": _default_model is not None,
        "cached_cities": cached_cities,
    }


@app.get("/api/geocode")
def geocode_city(city: str = Query(..., description="City name to search")):
    """
    Convert a city name to lat/lon using Open-Meteo's free geocoding API.
    Returns top 5 results.
    """
    url = (
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={city}&count=5&language=en&format=json"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return {"status": "not_found", "results": []}
        return {
            "status": "success",
            "results": [
                {
                    "name":    r.get("name"),
                    "country": r.get("country"),
                    "admin1":  r.get("admin1", ""),   # state/province
                    "lat":     r.get("latitude"),
                    "lon":     r.get("longitude"),
                }
                for r in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Geocoding failed: {e}")


@app.get("/api/current_weather")
def get_live_weather(
    lat: float = Query(default=DEFAULT_LAT),
    lon: float = Query(default=DEFAULT_LON),
):
    try:
        data = get_current_weather(lat=lat, lon=lon)
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch live weather: {e}")


@app.get("/api/train_status")
def train_status(
    lat: float = Query(...),
    lon: float = Query(...),
):
    """Check if a model is already cached for these coordinates."""
    entry = _get_cached(lat, lon)
    if entry:
        age_mins = (datetime.now() - entry["trained_at"]).total_seconds() / 60
        return {
            "cached":     True,
            "age_minutes": round(age_mins, 1),
            "mae":        round(entry["mae"], 4),
        }
    if _is_default_location(lat, lon) and _default_model:
        return {"cached": True, "age_minutes": 0, "mae": round(_default_mae, 4)}
    return {"cached": False}


@app.get("/api/train_and_predict")
def train_and_predict_stream(
    lat:          float = Query(...),
    lon:          float = Query(...),
    current_temp: float = Query(...),
    windspeed:    float = Query(...),
    humidity:     float = Query(...),
    cloudcover:   float = Query(...),
):
    """
    SSE endpoint — streams progress steps then the final forecast JSON.
    Client reads event: step / event: result / event: error
    """
    def event_stream():
        import json

        def send(event: str, data: dict):
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        try:
            # ── Step 1: Check cache / default model ──────────────────────────
            if _is_default_location(lat, lon) and _default_model:
                yield send("step", {"step": 1, "message": "Using pre-trained Greater Noida model…"})
                model, scaler, mae = _default_model, _default_scaler, _default_mae
            else:
                cached = _get_cached(lat, lon)
                if cached:
                    yield send("step", {"step": 1, "message": "Loading cached model for this city…"})
                    model, scaler, mae = cached["model"], cached["scaler"], cached["mae"]
                else:
                    # ── Step 1: Fetch historical data ────────────────────────
                    yield send("step", {"step": 1, "message": "Fetching 4 years of historical weather data…"})
                    df = fetch_historical_weather(lat, lon, years=4)

                    # ── Step 2: Train model ──────────────────────────────────
                    yield send("step", {"step": 2, "message": f"Training XGBoost model on {len(df)} days of data…"})
                    model, scaler, mae = train_for_city(df)
                    _set_cache(lat, lon, model, scaler, mae)

            # ── Step 3: Fetch recent lags + run forecast ─────────────────────
            yield send("step", {"step": 3, "message": "Fetching recent temperature history for forecast anchoring…"})

            # ── Step 4: Predict ──────────────────────────────────────────────
            yield send("step", {"step": 4, "message": "Running recursive 7-day forecast…"})
            forecast = _run_forecast(
                model, scaler, mae,
                current_temp, windspeed, humidity, cloudcover,
                lat, lon
            )

            yield send("result", {"forecast": forecast, "mae": round(mae, 4)})

        except Exception as e:
            yield send("error", {"message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
