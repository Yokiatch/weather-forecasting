# Setup
python -m venv venv
venv\Scripts\activate   # windows
pip install -r requirements.txt

# If data missing, place data/weather.csv (contains columns: date,temp,humidity,pressure,windspeed)
python scripts/train_model.py   # trains and creates models/
python scripts/evaluate_model.py  # saves results/performance_plot.png

# Run API
uvicorn app.main:app --reload

# Run Streamlit demo
streamlit run app/streamlit_app.py
