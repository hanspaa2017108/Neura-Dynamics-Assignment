import pytest


def test_answer_from_pdf_empty_retrieval(monkeypatch):
    import rag_pipeline.service as svc

    class DummyRetriever:
        def __init__(self, top_k: int = 4):
            self.top_k = top_k

        def retrieve(self, query: str):
            return []

    monkeypatch.setattr(svc, "QdrantRetriever", DummyRetriever)

    out = svc.answer_from_pdf("who is shah rukh khan?")
    assert out["route"] == "pdf"
    assert out["citations"] == []
    assert "couldn't find" in out["answer"].lower()


def test_answer_from_pdf_builds_citations_and_calls_llm(monkeypatch):
    import rag_pipeline.service as svc

    retrieved = [
        {"text": "Chunk A text", "page": 1, "chunk_ref": "refA"},
        {"text": "Chunk B text", "page": 2, "chunk_ref": "refB"},
    ]

    class DummyRetriever:
        def __init__(self, top_k: int = 4):
            self.top_k = top_k

        def retrieve(self, query: str):
            return retrieved

    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        def __init__(self, *args, **kwargs):
            self.invocations = []

        def invoke(self, messages, config=None):
            # record inputs for assertions
            self.invocations.append({"messages": messages, "config": config})
            return DummyMsg("ANSWER WITH CITATIONS (page=1, chunk_ref=refA)")

    monkeypatch.setattr(svc, "QdrantRetriever", DummyRetriever)
    monkeypatch.setattr(svc, "ChatOpenAI", lambda **kwargs: DummyLLM())

    out = svc.answer_from_pdf("test question")
    assert out["route"] == "pdf"
    assert out["answer"].startswith("ANSWER WITH CITATIONS")
    assert {"page": 1, "chunk_ref": "refA"} in out["citations"]
    assert {"page": 2, "chunk_ref": "refB"} in out["citations"]

