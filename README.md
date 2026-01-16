## Neura Dynamics – AI Engineer Assignment

This project is a **LangGraph agentic pipeline** with two capabilities:

- **Weather**: fetch real-time weather via **OpenWeatherMap** and summarize using an LLM.
- **PDF Q&A (RAG)**: answer questions from a PDF using **RAG** over **Qdrant Cloud** (embeddings + retrieval + grounded generation with citations).

It includes:
- **Streamlit** chat UI
- **LangSmith** tracing + evaluation-ready tagging
- **Unit tests** (pytest)

---

## Project structure

```
.
├── streamlit_app.py
├── requirements.txt
├── langgraph_pipeline/
│   ├── graph.py
│   ├── router.py
│   └── state.py
├── rag_pipeline/
│   ├── loader.py
│   ├── ingest.py
│   ├── retriever.py
│   └── service.py
├── openweather_pipeline/
│   ├── weather.py
│   └── service.py
├── scripts/test/
│   ├── test_qdrant_connection.py
│   ├── test_openweather_connection.py
│   └── test_langsmith_connection.py
└── tests/
    ├── test_router.py
    ├── test_weather_service.py
    ├── test_rag_service.py
    └── test_langgraph_graph.py
```

---

## Setup

### 1) Create and activate a virtualenv

```bash
python -m venv .venv
source .venv/bin/activate 
(venv or conda as per choice)
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment variables

Create a `.env` file in the repo root (do NOT commit it). Required:

```bash
# LLM
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4o-mini

# Weather
OPENWEATHER_API_KEY=...

# Qdrant Cloud
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION=neura-dynamics-assignment-v1
QDRANT_VECTOR_NAME=text

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=neura-dynamics-assignment
```

---

## Qdrant Cloud (collection configuration)

Create a collection in Qdrant Cloud:
- **Use case**: Global search
- **Search**: Simple Hybrid Search (supported), but retrieval currently uses **dense vectors** in vector field `text`
- **Dense vector name**: `text`
- **Dimensions**: `1536` (OpenAI `text-embedding-3-small`)
- **Distance**: Cosine

---

## One-time PDF ingestion (RAG index)

Place your PDF at:
- `data/test-rag-assignment.pdf`

Run ingestion (this upserts chunk embeddings into Qdrant Cloud):

```bash
python -m rag_pipeline.ingest
```

---

## Run the app (Streamlit)

```bash
streamlit run streamlit_app.py
```

The UI calls the LangGraph agent (`langgraph_pipeline/graph.py`) which routes:
- weather queries → OpenWeatherMap
- all other queries → PDF RAG

---

## LangSmith: tracing + evaluation

### Tracing
Tracing is enabled automatically when the LangSmith env vars are set.

### Evaluation-ready tagging
We tag runs so you can filter cleanly in LangSmith:
- **RAG answer generation**: tags `rag`, `eval_target`
- **Router LLM (only when used)**: tag `router`
- **Weather answer generation**: tag `weather`

### How to run evaluations (LangSmith UI)
- In LangSmith, filter runs by **Tag = `eval_target`** (RAG answer-generation step)
- Apply LLM-as-judge evaluators like:
  - **Answer Relevance**
  - **Hallucination** (as a groundedness proxy, evaluated against the provided context)

### Screenshots / artifacts (submission)
- **LangSmith dashboard / evaluations screenshots**: `<PASTE_GOOGLE_DRIVE_LINK_HERE>`
- **PDF used for RAG demo (Drive link)**: `<PASTE_GOOGLE_DRIVE_LINK_HERE>`

---

## Tests

### Unit tests (pytest)
These are **offline unit tests** that mock external services (no network calls):
- **Routing tests**: rule routing + LLM fallback behavior
- **Weather tests**: location parsing + error handling (OpenWeather mocked)
- **RAG tests**: empty retrieval behavior, citations formatting, and LLM invocation (Qdrant + LLM mocked)
- **LangGraph tests**: verifies the graph calls the correct branch (services mocked)

```bash
pytest
```

### Manual integration checks (real services)
These scripts hit real services using your `.env` keys (useful before deployment):

```bash
python scripts/test/test_qdrant_connection.py
python scripts/test/test_openweather_connection.py
python scripts/test/test_langsmith_connection.py
```

---

## Deployment notes

### Streamlit Community Cloud (simplest)
- Set the entrypoint to `streamlit_app.py`
- Add the env vars in Streamlit Cloud secrets
- Ensure Qdrant collection is already ingested (run `rag_pipeline.ingest` once beforehand)

---

## GitHub: push your code

Repo: `https://github.com/hanspaa2017108/Neura-Dynamics-Assignment.git` (wrap in backticks when sharing)

From the repo root:

```bash
git init
git add .
git commit -m "Initial implementation: LangGraph + RAG + Weather + Streamlit + LangSmith"
git branch -M main
git remote add origin https://github.com/hanspaa2017108/Neura-Dynamics-Assignment.git
git push -u origin main
```

If you already initialized git earlier, skip `git init` and just set the remote / push.

