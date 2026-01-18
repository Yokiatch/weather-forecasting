import requests

def get_current_weather(lat=28.6139, lon=77.2090):
    """
    Fetch live current weather using Open-Meteo API.
    Default location: Delhi (change if needed).
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current_weather=true"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    current = data["current_weather"]

    return {
        "temperature": current["temperature"],   # °C
        "windspeed": current["windspeed"]         # km/h
    }
