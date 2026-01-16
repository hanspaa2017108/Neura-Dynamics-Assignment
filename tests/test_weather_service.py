import pytest


def test_extract_location_of_pattern():
    from openweather_pipeline.service import _extract_location

    assert _extract_location("temperature of darjeeling") == "darjeeling"


def test_extract_location_of_of_pattern_prefers_rhs():
    from openweather_pipeline.service import _extract_location

    assert _extract_location("temp in sector 23 of nerul?") == "nerul"


def test_extract_location_strips_time_words():
    from openweather_pipeline.service import _extract_location

    assert _extract_location("should i carry umbrella in amritsar today?") == "amritsar"


def test_answer_from_weather_handles_notfound(monkeypatch):
    import openweather_pipeline.service as svc
    from pyowm.commons.exceptions import NotFoundError

    class DummyTool:
        def run(self, location: str):
            raise NotFoundError("Unable to find the resource")

    monkeypatch.setattr(svc, "WeatherTool", lambda: DummyTool())
    # Avoid LLM fallback in this test
    monkeypatch.setattr(svc, "_llm_extract_location", lambda q: None)

    out = svc.answer_from_weather("what's the weather in some made up place?")
    assert out["route"] == "weather"
    assert out["raw_weather"] is None
    assert "couldn't find that location" in out["answer"].lower()

