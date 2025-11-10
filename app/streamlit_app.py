import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Weather Forecast", page_icon="🌦", layout="centered")

st.title("🌤 Weather Forecasting App")
st.markdown("Enter the latest weather parameters below to predict the next day's temperature.")

model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1012.0)
windspeed = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 8.0)

st.markdown("### Temperature lags (previous 7 days)")
cols = st.columns(7)
temp_lags = []
for i in range(7):
    temp_lags.append(cols[i].number_input(f"Temp Lag {i+1} (°C)", 0.0, 50.0, 25.0))

if st.button("Predict Temperature"):
    data = {
        "humidity": [humidity],
        "pressure": [pressure],
        "windspeed": [windspeed],
    }
    for i in range(7):
        data[f"temp_lag{i+1}"] = [temp_lags[i]]
    
    df = pd.DataFrame(data)
    scaled = scaler.transform(df)
    pred = model.predict(scaled)
    
    st.success(f"🌡 Predicted Temperature: **{pred[0]:.2f}°C**")
