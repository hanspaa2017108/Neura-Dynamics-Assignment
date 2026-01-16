"""
RAG: Augmentation + Generation

Uses:
- QdrantRetriever (rag_pipeline.retriever) to fetch relevant PDF chunks
- OpenAI chat model via LangChain to generate an answer grounded in those chunks

Run:
  python -m rag_pipeline.answer

Optional env vars:
  OPENAI_CHAT_MODEL=gpt-4o-mini
  RAG_TOP_K=4
  QUERY="your question"
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_pipeline.retriever import QdrantRetriever

load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
TOP_K = int(os.getenv("RAG_TOP_K"))


def _format_context(retrieved: list[dict]) -> str:
    blocks: list[str] = []
    for i, r in enumerate(retrieved, start=1):
        page = r.get("page")
        chunk_ref = r.get("chunk_ref")
        text = (r.get("text") or "").strip()
        blocks.append(f"[{i}] page={page} chunk_ref={chunk_ref}\n{text}")
    return "\n\n".join(blocks)


def answer_question(query: str) -> str:
    retriever = QdrantRetriever(top_k=TOP_K)
    retrieved = retriever.retrieve(query)

    if not retrieved:
        return (
            "I couldn't find relevant information for that question in the ingested PDF. "
            "Try asking something covered by the document."
        )

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

    llm = ChatOpenAI(model=CHAT_MODEL)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "context": _format_context(retrieved)})


if __name__ == "__main__":
    q = os.getenv("QUERY", "What's the architecture of LLMs?")
    print(answer_question(q))

