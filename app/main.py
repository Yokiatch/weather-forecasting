from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.get("/")
def home():
    return {"message": "Weather Forecast API"}

@app.post("/predict/")
def predict_weather(data: dict):
    df = pd.DataFrame([data])
    scaled = scaler.transform(df)
    pred = model.predict(scaled)
    return {"predicted_temperature": round(float(pred[0]), 2)}
