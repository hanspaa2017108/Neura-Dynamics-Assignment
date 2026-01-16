# load pdf and create chunks

# rag_pipeline/loader.py

import hashlib
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _stable_chunk_id(source: str | None, page: int | None, text: str) -> str:
    """
    Stable chunk id for citations across re-runs (as long as chunk text is identical).
    """
    src = source or "unknown_source"
    pg = str(page) if page is not None else "unknown_page"
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{src}::p{pg}::{h}"


def _stable_point_uuid(chunk_ref: str) -> str:
    """
    Qdrant point IDs must be uint64 or UUID. We derive a deterministic UUID from the chunk ref.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_ref))


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Load a PDF and split it into semantic chunks.
    Each chunk retains page-level metadata.
    """

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    # Add stable ids:
    # - chunk_ref: readable, stable citation reference
    # - point_id: deterministic UUID to satisfy Qdrant point ID requirements
    for chunk in chunks:
        chunk_ref = _stable_chunk_id(
            source=chunk.metadata.get("source"),
            page=chunk.metadata.get("page"),
            text=chunk.page_content,
        )
        chunk.metadata["chunk_ref"] = chunk_ref
        chunk.metadata["point_id"] = _stable_point_uuid(chunk_ref)

    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_pdf("test-rag-assignment.pdf")
    print(f"Total chunks created: {len(chunks)}")
    print(chunks[0].metadata)
