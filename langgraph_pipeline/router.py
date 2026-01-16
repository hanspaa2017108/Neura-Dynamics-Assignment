import os
import re
from typing import Literal, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langgraph_pipeline.state import Route

load_dotenv()


_WEATHER_HINT_RE = re.compile(
    r"\b(weather|temperature|temp|rain|raining|forecast|humidity|wind|climate|umbrella|drizzle|storm|cloudy|sunny)\b",
    re.IGNORECASE,
)


def _rule_route(query: str) -> Tuple[Route | None, str | None]:
    q = query.strip()
    if not q:
        return None, None
    # Only route to weather if we see clear weather intent keywords.
    # Avoid treating generic phrases like "in which year" as a location.
    if _WEATHER_HINT_RE.search(q):
        return "weather", "rule_match(weather_keywords)"
    return None, None


def _llm_route(query: str) -> Tuple[Route, str]:
    model = os.getenv("OPENAI_ROUTER_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    llm = ChatOpenAI(model=model, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a routing classifier.\n"
                "Return ONLY one token: either 'weather' or 'pdf'.\n"
                "Choose 'weather' only if the user is asking about real-time weather conditions/forecast for a location.\n"
                "Choose 'pdf' for everything else (questions answered from the ingested PDF).\n",
            ),
            ("human", "{query}"),
        ]
    )

    route_str = (prompt | llm | StrOutputParser()).invoke(
        {"query": query},
        config={"tags": ["router"], "metadata": {"component": "router", "model": model}},
    ).strip().lower()
    route: Route = "weather" if "weather" in route_str else "pdf"
    return route, f"llm_router(model={model})"


def hybrid_route(query: str) -> Tuple[Route, str]:
    """
    Hybrid routing:
    - rules first (cheap + deterministic)
    - LLM fallback for ambiguous cases
    """
    route, reason = _rule_route(query)
    if route is not None:
        return route, reason or "rule_match"
    return _llm_route(query)

