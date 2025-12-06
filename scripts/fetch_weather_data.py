# scripts/fetch_weather_data.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("VC_API_KEY")
LOCATION = os.getenv("LOCATION", "Chennai,India")
START_DATE = os.getenv("START_DATE", "2023-01-01")
END_DATE = os.getenv("END_DATE", "2025-12-31")  # or "today"

if API_KEY is None:
    raise RuntimeError("Please set VC_API_KEY in .env")

# Visual Crossing timeline CSV endpoint
url = (
    f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    f"{LOCATION}/{START_DATE}/{END_DATE}"
    f"?unitGroup=metric&include=days&key={API_KEY}&contentType=csv"
)

print("Fetching data from Visual Crossing...")
resp = requests.get(url)
if resp.status_code != 200:
    raise RuntimeError(f"Failed to fetch data: HTTP {resp.status_code}\n{resp.text}")

os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", "weather.csv")
with open(csv_path, "wb") as f:
    f.write(resp.content)
print(f"Saved raw CSV to {csv_path}")

# Load and normalize columns to expected schema
df = pd.read_csv(csv_path)

# Visual Crossing may use 'datetime' for date, rename to 'date'
if 'datetime' in df.columns:
    df = df.rename(columns={'datetime': 'date'})

# Pick/rename columns to the ones your model expects: date,temp,humidity,pressure,windspeed
# Visual Crossing has 'temp', 'humidity', 'pressure', 'windspeed' (or similar).
expected = {}
for col in ['temp', 'humidity', 'pressure', 'windspeed']:
    if col in df.columns:
        expected[col] = col
# if pressure is not available as 'pressure', try 'sealevelpressure' or 'pressure' alternatives
if 'sealevelpressure' in df.columns and 'pressure' not in expected:
    df = df.rename(columns={'sealevelpressure': 'pressure'})

# Filter to required columns if present
keep_cols = ['date', 'temp', 'humidity', 'pressure', 'windspeed']
present = [c for c in keep_cols if c in df.columns]
df_out = df[present].copy()

# If windspeed is missing, try 'windspeed' alternatives
if 'windspeed' not in df_out.columns and 'windspd' in df.columns:
    df_out = df_out.rename(columns={'windspd': 'windspeed'})

# Save cleaned CSV (overwrites)
df_out.to_csv(csv_path, index=False)
print(f"Saved cleaned CSV with columns: {df_out.columns.tolist()} to {csv_path}")
