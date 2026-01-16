from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


Route = Literal["weather", "pdf"]


class AgentState(TypedDict):
    query: str
    route: NotRequired[Route]
    route_reason: NotRequired[str]
    result: NotRequired[dict[str, Any]]

