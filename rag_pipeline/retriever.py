# embed query and retrieve chunks from qdrant vector db

# rag_pipeline/retriever.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "neura-dynamics-assignment-v1")
VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "text")  # you configured this in Qdrant Cloud
MIN_SCORE = float(os.getenv("QDRANT_MIN_SCORE", "0.25"))


class QdrantRetriever:
    def __init__(self, top_k: int = 4):
        self.top_k = top_k

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

    def _query(self, query_vector: list[float]):
        """
        Support multiple qdrant-client versions.

        Newer versions expose `client.search(...)`.
        Some versions expose `client.query_points(...)`.
        """
        if hasattr(self.qdrant, "search"):
            return self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=(VECTOR_NAME, query_vector),
                limit=self.top_k,
            )

        if hasattr(self.qdrant, "query_points"):
            # query_points returns a response object containing `.points`
            return self.qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                using=VECTOR_NAME,
                limit=self.top_k,
            ).points

        raise AttributeError(
            "Your installed qdrant-client doesn't expose `search` or `query_points`. "
            "Upgrade it with: `pip install -U qdrant-client`."
        )

    def retrieve(self, query: str) -> list[dict]:
        query_vector = self.embeddings.embed_query(query)

        results = self._query(query_vector)

        formatted: list[dict] = []
        for r in results:
            payload = getattr(r, "payload", None) or {}
            score = getattr(r, "score", None)
            formatted.append(
                {
                    "score": score,
                    "text": payload.get("text", ""),
                    "page": payload.get("page"),
                    "chunk_ref": payload.get("chunk_ref"),
                    "source": payload.get("source"),
                }
            )
        # If the best match is still weak, treat as "not found in this document"
        scores = [x["score"] for x in formatted if isinstance(x.get("score"), (int, float))]
        if scores and max(scores) < MIN_SCORE:
            return []
        return [x for x in formatted if (x.get("score") is None or x["score"] >= MIN_SCORE)]


if __name__ == "__main__":
    retriever = QdrantRetriever()

    query = "What is architecture of LLMS?"
    results = retriever.retrieve(query)

    if not results:
        print(
            f"No relevant chunks found for query: {query!r}. "
            f"(This is expected if the question isn't answered by the ingested PDF.)"
        )

    for r in results:
        print(f"\nPage {r['page']} | chunk_ref={r['chunk_ref']} | score={r['score']}:")
        print(r["text"][:300], "...")
