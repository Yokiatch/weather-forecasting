Uncertainty-Aware Weather Forecasting System

Live Data Anchored, 7-Day Forecasting with Machine Learning & React

📌 Project Overview

This project is a machine learning–based weather forecasting system designed to predict the next 7 days of temperature using:

Historical learning
Live weather anchoring
Uncertainty estimation

Unlike traditional student projects that rely on static datasets or single-point predictions, this system:

Anchors forecasts to real-time weather data via Open-Meteo
Uses recursive multi-step forecasting to predict future days
Explicitly models prediction uncertainty using residual error analysis
Presents results through a decoupled FastAPI backend and a modern, glassmorphic React/Tailwind UI

The project is suitable for academic research, capstone evaluation, and demonstrating applied ML system design.

🎯 Problem Statement

Most beginner weather forecasting projects suffer from:

Reliance on static datasets that do not reflect real-world conditions
Providing single deterministic predictions without communicating reliability
Exposing internal ML features (e.g., lag variables) directly to users
Lack of system-level design (monolithic scripts instead of separated API and UI layers)

This project addresses these issues by building a realistic, interpretable, and modular forecasting system that combines machine learning with live data integration and uncertainty-aware predictions.

🧠 Key Concepts & Methodology
1️⃣ Historical Learning with Machine Learning
Core model: XGBoost Regressor
Trained on historical weather data
Uses lag-based features (previous 7 days’ temperatures) to capture temporal dependencies
2️⃣ Live Weather Anchoring (Open-Meteo API)
Backend fetches:
Temperature
Wind speed
Humidity
Cloud cover
These values act as Day 0 input for forecasting
Ensures predictions reflect current real-world conditions

⚠️ Live data is used for anchoring, not retraining

3️⃣ Recursive Multi-Step Forecasting (7-Day Prediction)
Predict Day 1
Feed prediction back to predict Day 2
Repeat recursively for 7 days

This is a standard and academically valid time-series approach.

4️⃣ Uncertainty-Aware Forecasting & Expanded Metrics

Instead of single-point predictions:

Residual errors are analyzed
Standard deviation is computed
A 95% confidence interval is generated
Uncertainty is dynamically calibrated based on deviation from current conditions

Additionally:

ML predictions are merged with 8-day API forecasts
Provides:
 1. Wind Speed
 2. Cloud Cover
 3. Precipitation

🏗️ System Architecture
       Open-Meteo API
             │
             ▼
    Live Weather Anchoring
             │
             ▼
 XGBoost ML Model (Historical)
             │
             ▼
  Recursive 7-Day Forecast
             │
             ▼
   Uncertainty Estimation
             │
             ▼
      FastAPI Backend  ──►  React / Tailwind Frontend

🧩 Project Structure
weather-forecasting/
│
├── app/
│   └── main.py              # FastAPI backend (prediction API & Data Merging)
│
├── frontend/                # React / Vite application
│   ├── src/
│   │   ├── App.jsx          # Main UI Component with Expandable Accordions
│   │   ├── index.css        # Tailwind v4 Configuration
│   │   └── main.jsx
│   └── package.json
│
├── scripts/
│   ├── train_model.py       # Model training + uncertainty computation
│   └── fetch_current_weather.py  
│
├── data/
│   └── weather.csv          # Historical training dataset
│
├── models/
│   ├── xgb_model.pkl        # Trained ML model
│   ├── scaler.pkl           # Feature scaler
│   └── uncertainty.pkl      # Residual-based uncertainty value
│
├── results/
│   ├── metrics.txt
│   └── actual_vs_pred.csv
│
├── requirements.txt         # Python dependencies
└── README.md

🚀 How to Run the Project
1️⃣ Python Backend Setup
pip install -r requirements.txt
pip install fastapi uvicorn pydantic requests
python scripts/train_model.py
uvicorn app.main:app --reload
API: http://localhost:8000
Docs: http://localhost:8000/docs

2️⃣ React Frontend Setup
cd frontend
npm install
npm run dev

UI: http://localhost:5173
📊 Output Example
Day 3 (Wednesday)

Predicted Temperature: 14.8 °C
Uncertainty: ±2.1 °C
Confidence Range: 12.7 – 16.9 °C

Expanded view:

Wind: 12 km/h
Clouds: 45%
Precipitation: 0.0 mm

📈 Evaluation Metrics
Mean Absolute Error (MAE)
Root Mean Square Error (RMSE)

Stored in:
results/metrics.txt

⚠️ Limitations
 1. Model does not retrain automatically
 2. Uncertainty reflects model confidence, not real-world guarantees
 3. Recursive forecasting may accumulate error

🔮 Future Enhancements
 1. Multi-variable forecasting (humidity, wind via ML)
 2. Deep learning models (LSTM / GRU)
 3. Automated retraining (CI/CD pipelines)
 4. Cloud deployment (AWS / GCP / Vercel)
    
