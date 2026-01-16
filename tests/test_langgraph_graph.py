def test_graph_routes_to_weather(monkeypatch):
    import langgraph_pipeline.graph as g

    monkeypatch.setattr(g, "answer_from_weather", lambda q: {"route": "weather", "answer": "OK"})
    monkeypatch.setattr(g, "answer_from_pdf", lambda q: {"route": "pdf", "answer": "NO"})

    out = g.run_agent("what's the weather in mumbai?")
    assert out["route"] == "weather"
    assert out["answer"] == "OK"


def test_graph_routes_to_pdf(monkeypatch):
    import langgraph_pipeline.graph as g

    monkeypatch.setattr(g, "answer_from_weather", lambda q: {"route": "weather", "answer": "NO"})
    monkeypatch.setattr(g, "answer_from_pdf", lambda q: {"route": "pdf", "answer": "OK"})

    out = g.run_agent("what is retrieval augmented generation?")
    assert out["route"] == "pdf"
    assert out["answer"] == "OK"

