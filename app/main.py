# app/main.py
from fastapi import FastAPI
import joblib, pandas as pd

app = FastAPI()
model = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    try:
        model = joblib.load('models/xgb_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Model and scaler loaded")
    except Exception as e:
        model = None
        scaler = None
        print(f"⚠️ Could not load model/scaler: {e}")

@app.get("/")
def home():
    return {"message": "Weather Forecast API", "model_loaded": model is not None}

@app.post("/predict/")
def predict_weather(data: dict):
    if model is None or scaler is None:
        return {"error": "Model unavailable. Run `python scripts/train_model.py` to create model."}
    df = pd.DataFrame([data])
    try:
        scaled = scaler.transform(df)
        pred = model.predict(scaled)
        return {"predicted_temperature": round(float(pred[0]), 2)}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
