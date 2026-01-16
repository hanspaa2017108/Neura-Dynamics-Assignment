import os
from dotenv import load_dotenv
from typing import Dict, Any

from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


class WeatherService:
    """
    Handles weather data retrieval using OpenWeatherMap.
    No LLM logic here.
    """

    def __init__(self):
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENWEATHER_API_KEY not set")

        self.client = OpenWeatherMapAPIWrapper(
            openweathermap_api_key=api_key
        )

    def get_weather(self, location: str) -> str:
        """
        Returns raw weather text from OpenWeatherMap.
        """
        return self.client.run(location)


class WeatherAnswerGenerator:
    """
    Uses LLM to convert raw weather data into a clean, friendly answer.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0,
        )

    def generate_answer(self, location: str, weather_text: str) -> str:
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant that summarizes weather information "
                    "clearly and concisely for users."
                )
            ),
            HumanMessage(
                content=(
                    f"Location: {location}\n"
                    f"Weather data:\n{weather_text}\n\n"
                    "Provide a clear and friendly weather summary."
                )
            ),
        ]

        # Tag weather generation runs for easy filtering in LangSmith (even if we don't evaluate them).
        response = self.llm.invoke(
            messages,
            config={
                "tags": ["weather"],
                "metadata": {"route": "weather", "component": "weather_answer_generation"},
            },
        )
        return response.content


class WeatherTool:
    """
    High-level weather interface used by LangGraph.
    """

    def __init__(self):
        self.weather_service = WeatherService()
        self.answer_generator = WeatherAnswerGenerator()

    def run(self, location: str) -> Dict[str, Any]:
        """
        End-to-end weather flow.
        """
        raw_weather = self.weather_service.get_weather(location)
        answer = self.answer_generator.generate_answer(location, raw_weather)

        return {
            "route": "weather",
            "location": location,
            "raw_weather": raw_weather,
            "answer": answer,
        }


if __name__ == "__main__":
    tool = WeatherTool()
    result = tool.run("Mumbai")

    print("\nWEATHER ANSWER:\n")
    print(result["answer"])