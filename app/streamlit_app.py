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

# ---------------- Dynamic Theming Logic ----------------
def apply_dynamic_theme(weather_code):
    # WMO Codes: 0-1 (Clear/Mostly Clear), 51-99 (Rain/Snow/Storm), Others (Cloudy/Fog)
    if weather_code <= 1:
        theme_name = "Sunny & Clear"
        bg_css = """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #ffe259, #ffa751);
        }
        </style>
        """
    elif 51 <= weather_code <= 99:
        theme_name = "Rainy / Stormy"
        bg_css = """
        <style>
        .stApp {
            background: linear-gradient(to bottom, #4b6cb7, #182848);
        }
        </style>
        """
    else:
        theme_name = "Cloudy / Overcast"
        bg_css = """
        <style>
        .stApp {
            background: linear-gradient(to right, #bdc3c7, #2c3e50);
        }
        </style>
        """
    st.markdown(bg_css, unsafe_allow_html=True)
    return theme_name

# ---------------- Header ----------------
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

# ---------------- Live Weather & Theming ----------------
st.subheader("🌡 Live Current Weather")

try:
    # Assuming get_current_weather() returns the expanded dictionary from our previous step
    live_weather = get_current_weather()
    current_temp = live_weather["temperature"]
    windspeed = live_weather["windspeed"]
    humidity = live_weather["humidity"]
    cloud_cover = live_weather["cloud_cover"]
    precipitation = live_weather["precipitation"]
    condition_code = live_weather["condition_code"]

    # Apply the theme based on the live code
    current_theme = apply_dynamic_theme(condition_code)
    st.info(f"**Current Condition:** {current_theme}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡 Temperature", f"{current_temp:.1f} °C")
    with col2:
        st.metric("💨 Wind Speed", f"{windspeed:.1f} km/h")
    with col3:
        st.metric("💧 Humidity", f"{humidity:.1f} %")

    col4, col5 = st.columns(2)
    with col4:
        st.metric("☁️ Cloud Cover", f"{cloud_cover:.1f} %")
    with col5:
        st.metric("🌧 Precipitation", f"{precipitation:.1f} mm")

except Exception:
    st.warning("⚠️ Live weather unavailable. Using manual input.")
    # Defaulting to sunny theme if API fails
    apply_dynamic_theme(0) 
    
    col1, col2 = st.columns(2)
    with col1:
        current_temp = st.number_input("Current Temperature (°C)", 0.0, 50.0, 25.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    with col2:
        windspeed = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 8.0)
        cloud_cover = st.number_input("Cloud Cover (%)", 0.0, 100.0, 20.0)

# ---------------- Prediction ----------------
if st.button("🔮 Predict Next 7 Days"):
    with st.spinner("Generating forecast..."):
        st.subheader("📅 7-Day Temperature Forecast")

        temp_lags = [current_temp] * 7
        predictions = []

        for day in range(1, 8):
            # Incorporating the new parameters into the model input
            data = {
                "humidity": humidity,
                "windspeed": windspeed,
                "cloudcover": cloud_cover, 
            }

            for i in range(7):
                data[f"temp_lag{i+1}"] = temp_lags[i]

            df = pd.DataFrame([data])
            # Note: Ensure your scaler and XGBoost model are retrained to accept 'cloud_cover' 
            # If they aren't retrained yet, comment out 'cloud_cover' in the data dict above for now.
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

        # -------- Trend Chart --------
        st.subheader("📈 7-Day Temperature Trend")

        days = [f"Day {p['day']}" for p in predictions]
        temps = [p["temp"] for p in predictions]
        lows = [p["low"] for p in predictions]
        highs = [p["high"] for p in predictions]

        fig, ax = plt.subplots()
        ax.plot(days, temps, marker="o", color="#e74c3c", label="Predicted Temperature")
        ax.fill_between(days, lows, highs, color="#e74c3c", alpha=0.2, label="95% Confidence Interval")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Forecast with Uncertainty Bounds")
        
        # Setting a transparent background for the chart to match the dynamic Streamlit background
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
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