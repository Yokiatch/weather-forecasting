import requests

def get_current_weather():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=28.4744"
        "&longitude=77.5040"
        "&current=temperature_2m,windspeed_10m"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()["current"]
    return {
        "temperature": data["temperature_2m"],
        "windspeed": data["windspeed_10m"]
    }
