import sys
import os

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from scripts.fetch_current_weather import get_current_weather

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="7-Day Weather Forecast",
    page_icon="🌦",
    layout="centered"
)

st.title("🌤 7-Day Weather Forecast")
st.markdown(
    "Machine learning–based **7-day temperature forecasting** "
    "anchored to **live weather data** with **uncertainty-aware predictions**."
)

st.caption("📍 Location: Greater Noida, India")

# ---------------- Load ML Artifacts ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    uncertainty = joblib.load("models/uncertainty.pkl")
    return model, scaler, uncertainty

model, scaler, uncertainty = load_artifacts()

# ---------------- Live Weather ----------------
st.subheader("🌡 Live Current Weather (Open-Meteo)")

try:
    live_weather = get_current_weather()
    current_temp = live_weather["temperature"]
    windspeed = live_weather["windspeed"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌡 Temperature", f"{current_temp:.1f} °C")
    with col2:
        st.metric("💨 Wind Speed", f"{windspeed:.1f} km/h")

except Exception:
    st.warning("⚠️ Live weather unavailable. Using manual input.")
    current_temp = st.number_input("Current Temperature (°C)", 0.0, 50.0, 25.0)
    windspeed = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 8.0)

# ---------------- Other Inputs ----------------
st.subheader("🌬 Atmospheric Conditions")
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)

# ---------------- Prediction ----------------
if st.button("🔮 Predict Next 7 Days"):
    with st.spinner("Generating forecast..."):
        st.subheader("📅 7-Day Temperature Forecast")

        temp_lags = [current_temp] * 7
        predictions = []

        for day in range(1, 8):
            data = {
                "humidity": humidity,
                "windspeed": windspeed,
            }

            for i in range(7):
                data[f"temp_lag{i+1}"] = temp_lags[i]

            df = pd.DataFrame([data])
            scaled = scaler.transform(df)
            pred = model.predict(scaled)[0]

            # -------- Uncertainty Estimation --------
            temp_diff = abs(pred - current_temp)
            dynamic_uncertainty = uncertainty * (1 + temp_diff / 10)

            margin = 1.96 * dynamic_uncertainty
            low = pred - margin
            high = pred + margin

            predictions.append({
                "day": day,
                "temp": pred,
                "low": low,
                "high": high,
                "uncertainty": margin
            })

            temp_lags = [pred] + temp_lags[:-1]

        # -------- Display Forecast Cards --------
        for p in predictions:
            st.success(
                f"**Day {p['day']}**\n\n"
                f"🌡 **Predicted Temperature:** {p['temp']:.2f} °C\n\n"
                f"📊 **Uncertainty:** ±{p['uncertainty']:.2f} °C\n\n"
                f"🔍 **Confidence Range:** {p['low']:.2f} °C – {p['high']:.2f} °C"
            )

        # -------- Trend Chart --------
        st.subheader("📈 7-Day Temperature Trend")

        days = [f"Day {p['day']}" for p in predictions]
        temps = [p["temp"] for p in predictions]
        lows = [p["low"] for p in predictions]
        highs = [p["high"] for p in predictions]

        fig, ax = plt.subplots()
        ax.plot(days, temps, marker="o", label="Predicted Temperature")
        ax.fill_between(days, lows, highs, alpha=0.3, label="Uncertainty Range")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("7-Day Forecast with Uncertainty")
        ax.legend()

        st.pyplot(fig)

        st.caption(
            "ℹ️ Forecast combines historical learning with live observations. "
            "Uncertainty represents model confidence, not exact bounds."
        )

# ---------------- Explanation ----------------
with st.expander("ℹ️ How this forecast works"):
    st.write(
        """
        - The model is trained on **historical observed weather data**.
        - Live weather data is fetched from **Open-Meteo** to anchor predictions.
        - Future temperatures are generated using **recursive forecasting**.
        - Uncertainty is estimated using **residual error analysis**.
        """
    )
