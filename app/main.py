# app/main.py
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

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


@app.post("/predict/")
def predict_weather(data: dict):
    if model is None or scaler is None or uncertainty is None:
        return {
            "error": "Model unavailable. Please run `python scripts/train_model.py` first."
        }

    try:
        df = pd.DataFrame([data])
        scaled = scaler.transform(df)
        pred = model.predict(scaled)

        temp = float(pred[0])

        # 95% confidence interval
        margin = 1.96 * uncertainty
        lower = temp - margin
        upper = temp + margin

        return {
            "predicted_temperature": round(temp, 2),
            "confidence_interval": f"±{round(margin, 2)} °C",
            "range": [
                round(lower, 2),
                round(upper, 2)
            ]
        }

    except Exception as e:
        return {
            "error": f"Prediction failed: {e}"
        }
