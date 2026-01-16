import types

import pytest


def test_rule_routes_weather_keywords():
    from langgraph_pipeline.router import hybrid_route

    route, reason = hybrid_route("should I carry an umbrella in Mumbai today?")
    assert route == "weather"
    assert "rule_match" in (reason or "")


def test_rule_does_not_route_generic_in_which_year():
    from langgraph_pipeline.router import hybrid_route

    # No weather keywords → should not be rule-routed to weather
    route, reason = hybrid_route("chain of thought was demonstrated in which year?")
    assert route in {"pdf", "weather"}  # LLM fallback decides; rule should NOT force weather
    assert "rule_match" not in (reason or "")


def test_llm_fallback_can_route_pdf(monkeypatch):
    import langgraph_pipeline.router as router_mod

    def fake_llm_route(query: str):
        return "pdf", "llm_router(fake)"

    monkeypatch.setattr(router_mod, "_llm_route", fake_llm_route)

    route, reason = router_mod.hybrid_route("tell me about transformers")
    assert route == "pdf"
    assert reason == "llm_router(fake)"

