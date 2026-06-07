import requests
import time

def get_current_weather(lat: float = 28.4744, lon: float = 77.5040):
    """
    Fetch current weather from Open-Meteo.
    Accepts optional lat/lon for geolocation support (defaults to Greater Noida).
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&current=temperature_2m,wind_speed_10m,relative_humidity_2m,cloud_cover,precipitation,weather_code"
    )

    last_exception = None
    for attempt in range(3):  # retry up to 3 times on rate-limit
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 429:
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
                continue
            response.raise_for_status()  # raises on 4xx/5xx (excluding 429 handled above)
            data = response.json()["current"]
            return {
                "temperature": data["temperature_2m"],
                "windspeed":   data["wind_speed_10m"],
                "humidity":    data["relative_humidity_2m"],
                "cloud_cover": data["cloud_cover"],
                "precipitation": data["precipitation"],
                "condition_code": data["weather_code"],
            }
        except requests.RequestException as e:
            last_exception = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to fetch weather after 3 attempts: {last_exception}")
