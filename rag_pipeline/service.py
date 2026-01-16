"""
RAG service wrapper (single entrypoint)

Goal:
Expose a clean, side-effect-free function that LangGraph + Streamlit can call.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_pipeline.retriever import QdrantRetriever

load_dotenv()


def answer_from_pdf(query: str) -> dict[str, Any]:
    """
    Answer a question using RAG over the ingested PDF collection in Qdrant.

    Returns a structured dict so callers (LangGraph/Streamlit/tests) can easily consume it.
    """
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    top_k = int(os.getenv("RAG_TOP_K", "4"))

    retriever = QdrantRetriever(top_k=top_k)
    retrieved = retriever.retrieve(query)

    citations: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for r in retrieved:
        key = (r.get("page"), r.get("chunk_ref"))
        if key in seen:
            continue
        seen.add(key)
        citations.append({"page": r.get("page"), "chunk_ref": r.get("chunk_ref")})

    if not retrieved:
        return {
            "route": "pdf",
            "query": query,
            "answer": (
                "I couldn't find relevant information for that question in the ingested PDF. "
                "Try asking something covered by the document."
            ),
            "citations": [],
        }

    context_blocks: list[str] = []
    for i, r in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[{i}] page={r.get('page')} chunk_ref={r.get('chunk_ref')}\n{(r.get('text') or '').strip()}"
        )
    context = "\n\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant.\n"
                "Answer ONLY using the provided context.\n"
                "If the answer is not in the context, say you don't know.\n"
                "You MUST include citations in the final answer using (page, chunk_ref) from the context items.\n",
            ),
            (
                "human",
                "Question:\n{query}\n\n"
                "Context:\n{context}\n\n"
                "Write the answer and include citations like: (page=7, chunk_ref=...).",
            ),
        ]
    )

    # Use direct llm.invoke so it's easy to unit-test and we still get full prompt/context in traces.
    messages = prompt.format_messages(query=query, context=context)
    llm = ChatOpenAI(model=chat_model, temperature=0)
    response = llm.invoke(
        messages,
        config={
            "tags": ["rag", "eval_target"],
            "metadata": {
                "route": "pdf",
                "component": "rag_answer_generation",
                "top_k": top_k,
                "model": chat_model,
            },
        },
    )
    answer = getattr(response, "content", None) or str(response)

    return {
        "route": "pdf",
        "query": query,
        "answer": answer,
        "citations": citations,
    }


if __name__ == "__main__":
    q = os.getenv("QUERY", "what is transformers??")
    result = answer_from_pdf(q)
    print(result["answer"])
    print("\nCITATIONS:", result["citations"])

