import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_weather(lat: float, lon: float, years: int = 4) -> pd.DataFrame:
    """
    Fetch historical daily weather from Open-Meteo archive API.
    Returns a DataFrame with columns: datetime, temp, windspeed, humidity, cloudcover
    """
    end_date   = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")  # archive has ~5 day lag
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_mean,wind_speed_10m_max,relative_humidity_2m_mean,cloud_cover_mean,precipitation_sum"
        "&timezone=auto"
    )

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    df = pd.DataFrame({
        "datetime":   daily["time"],
        "temp":       daily["temperature_2m_mean"],
        "windspeed":  daily["wind_speed_10m_max"],
        "humidity":   daily["relative_humidity_2m_mean"],
        "cloudcover": daily["cloud_cover_mean"],
        "precip":     daily["precipitation_sum"],
    })

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.dropna().sort_values("datetime").reset_index(drop=True)
    return df


def fetch_recent_7days(lat: float, lon: float) -> list[float]:
    """
    Fetch last 7 days of mean temperature for lag initialization.
    Uses Open-Meteo forecast API with past_days parameter.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean"
        "&past_days=7&forecast_days=1"
        "&timezone=auto"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    temps = resp.json()["daily"]["temperature_2m_mean"]
    # Return last 7 values (most recent last), reversed for lag order
    return list(reversed(temps[-7:]))
