Weather Forecasting System
Live Data Anchored, Uncertainty-Aware, 7-Day Temperature Forecasting
📌 Project Overview

This project is a machine learning–based weather forecasting system designed to predict the next 7 days of temperature using historical learning, live weather anchoring, and uncertainty estimation.

Unlike traditional student projects that rely on static datasets or single-point predictions, this system:

Anchors forecasts to real-time weather data

Uses recursive multi-step forecasting to predict future days

Explicitly models prediction uncertainty using residual error analysis

Presents results through a FastAPI backend and a Streamlit-based UI

The project is suitable for academic research, capstone evaluation, and demonstration of applied ML system design.

🎯 Problem Statement

Most beginner weather forecasting projects suffer from the following limitations:

Reliance on static datasets that do not reflect real-world conditions

Providing single deterministic predictions without communicating reliability

Exposing internal ML features (e.g., lag variables) directly to users

Lack of system-level design (API, UI, deployment considerations)

This project addresses these issues by building a realistic, interpretable, and modular forecasting system that combines machine learning with live data integration and uncertainty-aware predictions.

🧠 Key Concepts & Methodology
1️⃣ Historical Learning with Machine Learning

The core prediction model is an XGBoost Regressor

Trained on historical weather data

Uses lag-based features (previous 7 days’ temperatures) to capture temporal dependencies

2️⃣ Live Weather Anchoring (Open-Meteo API)

The system fetches current temperature and wind speed from the Open-Meteo API

These live values are used as the starting point (Day 0) for forecasting

This ensures predictions reflect current real-world conditions, not just historical averages

⚠️ Note: Live data is used for anchoring, not for retraining the model in real time.

3️⃣ Recursive Multi-Step Forecasting (7-Day Prediction)

The model predicts Day 1 temperature

That prediction is fed back as input to predict Day 2

This process is repeated recursively for 7 days

This approach is widely used in time-series forecasting and is academically valid.

4️⃣ Uncertainty-Aware Forecasting (Key Research Feature)

Instead of providing only a single temperature value, the system estimates prediction uncertainty:

Residual errors from training are analyzed

Standard deviation of residuals is computed

A 95% confidence interval is generated for each prediction

To improve realism, uncertainty is dynamically calibrated based on deviation from current conditions.

Each forecast is presented as:

Predicted Temperature ± Uncertainty
Confidence Range (Lower – Upper)


This makes the system interpretable and trustworthy, which is a key research contribution.

🏗️ System Architecture
Open-Meteo API
      │
      ▼
Live Weather Anchoring
      │
      ▼
XGBoost ML Model (Historical Learning)
      │
      ▼
Recursive 7-Day Forecast
      │
      ▼
Uncertainty Estimation
      │
      ▼
FastAPI Backend  ──► Streamlit Frontend

🧩 Project Structure
weather-forecasting/
│
├── app/
│   ├── main.py              # FastAPI backend (prediction API)
│   └── streamlit_app.py     # Streamlit UI (7-day forecast)
│
├── scripts/
│   ├── train_model.py       # Model training + uncertainty computation
│   ├── fetch_current_weather.py  # Live weather fetch (Open-Meteo)
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
├── requirements.txt
└── README.md

🚀 How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the model
python scripts/train_model.py


This generates:

Trained model

Scaler

Uncertainty parameter

3️⃣ Run FastAPI backend
uvicorn app.main:app --reload


API will be available at:

http://127.0.0.1:8000

4️⃣ Run Streamlit UI
streamlit run app/streamlit_app.py

📊 Output Example

For each day, the system displays:

Predicted Temperature

Uncertainty (± °C)

Confidence Range

Example:

Day 3
Predicted Temperature: 14.8 °C
Uncertainty: ±2.1 °C
Confidence Range: 12.7 – 16.9 °C

📈 Evaluation Metrics

The model is evaluated using:

Mean Absolute Error (MAE)

Root Mean Square Error (RMSE)

These metrics are stored in:

results/metrics.txt

🔬 Research Contributions

This project introduces the following research-oriented elements:

Live data anchoring using an external weather API

Recursive multi-step time-series forecasting

Residual-based uncertainty estimation

Dynamic uncertainty calibration

Separation of ML logic, API layer, and UI

These features go beyond basic ML demos and are suitable for academic discussion.

⚠️ Limitations (Important for Research Papers)

The model is trained on historical data and does not retrain automatically

Uncertainty represents model confidence, not meteorological guarantees

Only temperature is forecasted (other variables can be extended)

Recursive forecasting can accumulate error over longer horizons

These limitations are explicitly acknowledged and discussed in the research paper.

🔮 Future Enhancements

Multi-variable forecasting (humidity, rainfall, wind)

Deep learning models (LSTM / GRU)

Automated retraining pipelines

Cloud deployment (AWS / GCP / Azure)

Location-based dynamic forecasting

📄 For Research Paper Generation (Important)

This README is intentionally written so that an AI model can generate a complete research paper using it alone.

It contains:

Problem definition

Methodology

Architecture

Results interpretation

Limitations

Research contributions

🧠 Key Takeaway

This project demonstrates how machine learning, live data integration, and uncertainty modeling can be combined to build a realistic and interpretable weather forecasting system suitable for academic and real-world exploration.