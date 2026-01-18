import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import pandas as pd

from scripts.fetch_current_weather import get_current_weather

st.set_page_config(page_title="7-Day Weather Forecast", page_icon="🌦", layout="centered")

st.title("🌤 7-Day Weather Forecast")
st.markdown("Predict the **next 7 days temperature** with **uncertainty awareness**.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    uncertainty = joblib.load("models/uncertainty.pkl")
    return model, scaler, uncertainty

model, scaler, uncertainty = load_artifacts()

# --- User inputs (REALISTIC) ---
st.subheader("Current Weather Conditions")

try:
    live_weather = get_current_weather()
    current_temp = live_weather["temperature"]
    windspeed = live_weather["windspeed"]

    st.subheader("🌡 Live Current Weather")
    st.metric("Current Temperature", f"{current_temp} °C")
    st.metric("Wind Speed", f"{windspeed} km/h")

except Exception as e:
    st.error("⚠️ Could not fetch live weather. Using manual input.")
    current_temp = st.number_input("Current Temperature (°C)", 0.0, 50.0, 25.0)
    windspeed = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 8.0)

humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1012.0)


if st.button("Predict Next 7 Days"):
    st.subheader("📅 7-Day Forecast")

    # Initialize lags with current temperature
    temp_lags = [current_temp] * 7
    predictions = []

    for day in range(1, 8):
        data = {
            "humidity": humidity,
            "pressure": pressure,
            "windspeed": windspeed,
        }

        for i in range(7):
            data[f"temp_lag{i+1}"] = temp_lags[i]

        df = pd.DataFrame([data])
        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]

        # Dynamic uncertainty calibration
        temp_diff = abs(pred - current_temp)
        dynamic_uncertainty = uncertainty * (1 + temp_diff / 10)

        margin = 1.96 * dynamic_uncertainty


        predictions.append({
            "day": day,
            "temp": pred,
            "low": pred - margin,
            "high": pred + margin
        })

        # shift lags
        temp_lags = [pred] + temp_lags[:-1]

    # Display results
    for p in predictions:
        st.success(
            f"Day {p['day']}: "
            f"{p['temp']:.2f}°C  "
            f"(Range: {p['low']:.2f}°C – {p['high']:.2f}°C)"
        )
