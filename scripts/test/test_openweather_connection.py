import os
from dotenv import load_dotenv

from langchain_community.utilities import OpenWeatherMapAPIWrapper

load_dotenv()

def test_openweather(city: str):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY not found in environment")

    weather = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=api_key
    )

    result = weather.run(city)
    return result


if __name__ == "__main__":
    city = "Mumbai"
    print(f"Testing OpenWeatherMap API for city: {city}\n")

    response = test_openweather(city)

    print("Raw response:")
    print(response)
