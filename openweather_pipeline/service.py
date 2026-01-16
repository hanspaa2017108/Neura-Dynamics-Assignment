"""
Weather service wrapper (single entrypoint)

Goal:
Expose a clean function that LangGraph + Streamlit can call.

Run:
  python -m openweather_pipeline.service

Env vars:
  OPENWEATHER_API_KEY
  OPENAI_CHAT_MODEL=gpt-4o-mini   (optional)
  QUERY="What's the weather in Mumbai?" (optional)
"""

import os
import re
import json
from typing import Any

from dotenv import load_dotenv
from pyowm.commons.exceptions import NotFoundError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from openweather_pipeline.weather import WeatherTool

load_dotenv()


_CITY_RE = re.compile(
    # Allow numbers/commas so we can capture phrases like:
    # - "sector 23 of nerul"
    # - "Bengaluru, IN"
    # - "temperature of darjeeling"
    r"\b(?:in|at|for|of)\s+([A-Za-z0-9][A-Za-z0-9 ,.'-]{1,80})\b",
    re.IGNORECASE,
)

_GENERIC_PLACE_TOKENS = {
    "sector",
    "block",
    "area",
    "district",
    "zone",
    "ward",
    "locality",
    "city",
    "town",
    "village",
}

_TRAILING_TIME_TOKENS = {
    "today",
    "now",
    "tonight",
    "tomorrow",
    "morning",
    "afternoon",
    "evening",
}


def _extract_location(query: str) -> str | None:
    """
    Best-effort location extraction. Keeps it lightweight and deterministic.
    """
    m = _CITY_RE.search(query)
    if not m:
        return None
    loc = m.group(1).strip()
    # Trim trailing punctuation
    loc = loc.rstrip("?.!,;:")

    # If user wrote "X of Y", prefer Y (e.g., "sector 23 of nerul" -> "nerul")
    if " of " in loc.lower():
        loc = loc.split(" of ", 1)[1].strip()

    # Strip common trailing time words (e.g., "amritsar today" -> "amritsar")
    tokens = [t for t in re.split(r"\s+", loc) if t]
    while tokens and tokens[-1].lower() in _TRAILING_TIME_TOKENS:
        tokens.pop()
    if len(tokens) >= 2 and tokens[-2].lower() == "this" and tokens[-1].lower() in _TRAILING_TIME_TOKENS:
        tokens = tokens[:-2]
    loc = " ".join(tokens).strip()
    return loc or None


def _location_candidates(query: str) -> list[str]:
    """
    Generate a small set of increasingly-generic location candidates.
    This prevents crashes when the user provides a neighborhood + city that OWM can't resolve.
    """
    loc = _extract_location(query)
    if not loc:
        return []

    candidates = [loc]

    # If it's a multi-part location, try the last 1-2 tokens (often the city).
    tokens = [t for t in re.split(r"\s+", loc) if t]
    if len(tokens) >= 2:
        last = tokens[-1]
        last2 = " ".join(tokens[-2:])
        if last.lower() not in _GENERIC_PLACE_TOKENS:
            candidates.append(last)
        if tokens[-2].lower() not in _GENERIC_PLACE_TOKENS and last.lower() not in _GENERIC_PLACE_TOKENS:
            candidates.append(last2)

    # If it looks like a city name, also try adding country hint (helps OWM resolution).
    # Keep this conservative to avoid making things worse for non-India queries.
    if len(tokens) >= 1:
        city = tokens[-1]
        if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,64}", city):
            candidates.append(f"{city}, IN")

    # De-dupe while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        c2 = c.strip()
        if c2 and c2 not in seen:
            seen.add(c2)
            out.append(c2)
    return out


def _llm_extract_location(query: str) -> str | None:
    """
    LLM fallback: extract a clean location string for OpenWeatherMap.
    Returns None if no location is present.
    """
    model = os.getenv("OPENAI_LOCATION_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    llm = ChatOpenAI(model=model, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract the location from the user's weather question.\n"
                "Return ONLY valid JSON with this schema:\n"
                '{{ "location": string | null }}\n'
                'If no location is present, return {{"location": null}}.\n'
                "The location should be a city/region string suitable for OpenWeatherMap.\n"
                "Do not include time words like today/now/tonight.\n",
            ),
            ("human", "{query}"),
        ]
    )

    raw = (prompt | llm | StrOutputParser()).invoke({"query": query}).strip()
    try:
        data = json.loads(raw)
    except Exception:
        return None

    loc = data.get("location")
    if not isinstance(loc, str):
        return None
    loc = loc.strip().rstrip("?.!,;:")
    return loc or None


def answer_from_weather(query: str) -> dict[str, Any]:
    """
    Answer a weather query using OpenWeatherMap + LLM summarization.
    Returns structured output for callers.
    """
    tool = WeatherTool()
    candidates = _location_candidates(query)
    if not candidates:
        # Weather intent is clear but we couldn't parse a location deterministically.
        # Use LLM extraction before asking the user to rephrase.
        llm_loc = _llm_extract_location(query)
        if llm_loc:
            candidates = [llm_loc]
        else:
            return {
                "route": "weather",
                "query": query,
                "location": None,
                "answer": "Please provide a location, e.g. 'What's the weather in Mumbai?'",
                "raw_weather": None,
            }

    last_err: Exception | None = None
    for location in candidates:
        try:
            result = tool.run(location)
            break
        except NotFoundError as e:
            last_err = e
    else:
        # All rule-derived candidates failed; try LLM extraction as a fallback.
        llm_loc = _llm_extract_location(query)
        if llm_loc:
            try:
                result = tool.run(llm_loc)
            except NotFoundError as e:
                last_err = e
            else:
                return {
                    "route": "weather",
                    "query": query,
                    "location": llm_loc,
                    "answer": result.get("answer"),
                    "raw_weather": result.get("raw_weather"),
                    "route_reason": "llm_location_fallback",
                }

        # Still failed
        return {
            "route": "weather",
            "query": query,
            "location": candidates[0],
            "answer": (
                "I couldn't find that location in OpenWeatherMap. "
                "Try a city name like 'Amritsar' or 'Amritsar, IN'."
            ),
            "raw_weather": None,
            "error": str(last_err) if last_err else "NotFoundError",
        }

    # Normalize output shape for LangGraph/Streamlit
    return {
        "route": "weather",
        "query": query,
        "location": result.get("location", candidates[0]),
        "answer": result.get("answer"),
        "raw_weather": result.get("raw_weather"),
    }


if __name__ == "__main__":
    q = os.getenv("QUERY", "What's the weather in Hebbal?")
    out = answer_from_weather(q)
    print(out["answer"])

