from __future__ import annotations

from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, END

from langgraph_pipeline.router import hybrid_route
from langgraph_pipeline.state import AgentState, Route
from openweather_pipeline.service import answer_from_weather
from rag_pipeline.service import answer_from_pdf


def route_node(state: AgentState) -> AgentState:
    query = state["query"]
    route, reason = hybrid_route(query)
    return {**state, "route": route, "route_reason": reason}


def weather_node(state: AgentState) -> AgentState:
    result = answer_from_weather(state["query"])
    return {**state, "result": result}


def pdf_node(state: AgentState) -> AgentState:
    result = answer_from_pdf(state["query"])
    return {**state, "result": result}


def _branch(state: AgentState) -> Route:
    return state["route"]


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("weather", weather_node)
    g.add_node("pdf", pdf_node)

    g.set_entry_point("route")
    g.add_conditional_edges("route", _branch, {"weather": "weather", "pdf": "pdf"})
    g.add_edge("weather", END)
    g.add_edge("pdf", END)
    return g.compile()


def run_agent(query: str) -> dict[str, Any]:
    app = build_graph()
    out: Dict[str, Any] = app.invoke({"query": query})
    # Normalize
    result = out.get("result") or {}
    return {
        "query": query,
        "route": out.get("route"),
        "route_reason": out.get("route_reason"),
        **result,
    }


if __name__ == "__main__":
    import os

    q = os.getenv("QUERY", "What's the weather in Mumbai?")
    out = run_agent(q)
    # Friendly CLI output (Streamlit will render these fields directly)
    print(out.get("answer", ""))
    if out.get("route"):
        print(f"\n[route] {out.get('route')} ({out.get('route_reason')})")
    if out.get("route") == "pdf" and out.get("citations"):
        print("\n[citations]")
        for c in out["citations"]:
            print(f"- page={c.get('page')} chunk_ref={c.get('chunk_ref')}")

