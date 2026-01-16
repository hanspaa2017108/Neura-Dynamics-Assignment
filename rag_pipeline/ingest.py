# generate embeddings and store in qdrant vector db

# rag_pipeline/ingest.py

import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_openai import OpenAIEmbeddings

from rag_pipeline.loader import load_and_chunk_pdf

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
PDF_PATH = os.getenv("PDF_PATH", "data/test-rag-assignment.pdf")
VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "text")  # you configured this in Qdrant Cloud
UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "16"))
QDRANT_TIMEOUT_SECONDS = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "120"))


def ingest_pdf():
    # Load + chunk PDF
    chunks = load_and_chunk_pdf(PDF_PATH)

    # Embedding model (LOCKED)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # 1536 dims
    )

    # Qdrant client
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=QDRANT_TIMEOUT_SECONDS,
    )

    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        point_id = chunk.metadata.get("point_id")
        chunk_ref = chunk.metadata.get("chunk_ref")
        point = PointStruct(
            # Use deterministic UUID so re-ingestion overwrites cleanly (and Qdrant accepts the ID)
            id=str(point_id),
            vector={VECTOR_NAME: vector},
            payload={
                "text": chunk.page_content,
                "page": chunk.metadata.get("page"),
                "source": chunk.metadata.get("source"),
                # human-readable ref for citations
                "chunk_ref": chunk_ref,
            },
        )
        points.append(point)

    total = 0
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
        total += len(batch)

    print(f"Ingested {total} chunks into Qdrant collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    ingest_pdf()
