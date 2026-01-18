import pandas as pd

df = pd.read_csv("data/weather.csv")

# Rename datetime → date
if "datetime" in df.columns:
    df.rename(columns={"datetime": "date"}, inplace=True)

# Columns that actually exist
required_cols = ["date", "temp", "humidity", "windspeed"]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required_cols]

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Drop missing values
df = df.dropna()

# Sort
df = df.sort_values("date")

df.to_csv("data/weather.csv", index=False)
print("✅ weather.csv cleaned (without pressure)")
