import requests

def get_current_weather():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=28.4744"
        "&longitude=77.5040"
        # Appended new parameters, including weather_code for UI logic
        "&current=temperature_2m,wind_speed_10m,relative_humidity_2m,cloud_cover,precipitation,weather_code"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()["current"]
    
    return {
        "temperature": data["temperature_2m"],
        "windspeed": data["wind_speed_10m"],
        "humidity": data["relative_humidity_2m"],
        "cloud_cover": data["cloud_cover"],
        "precipitation": data["precipitation"],
        "condition_code": data["weather_code"] # We will use this for the Streamlit UI
    }