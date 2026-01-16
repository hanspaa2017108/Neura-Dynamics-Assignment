import os

import streamlit as st
from dotenv import load_dotenv

from langgraph_pipeline.graph import run_agent

load_dotenv()

st.set_page_config(page_title="Neura Dynamics Assignment Demo", page_icon="🤖", layout="centered")

st.title("Neura Dynamics Assignment Demo")
st.caption("Weather (OpenWeatherMap) + PDF Q&A (RAG on Qdrant) via LangGraph")

QUICK_QUESTIONS = [
    # Weather
    "Weather — What's the weather in Mumbai right now?",
    "Weather — Should I carry an umbrella in Pune today?",
    "Weather — Temperature of Darjeeling?",
    "Weather — Is it raining in Delhi right now?",
    "Weather — What's the humidity in Bengaluru?",
    "Weather — What's the wind speed in Chennai today?",
    "Weather — Forecast for Hyderabad today?",
    "Weather — What's the weather in Amritsar, IN today?",
    "Weather — What's the temperature in Nerul?",
    "Weather — What's the temperature in sector 23 of Nerul?",
    # PDF
    "PDF — What is the main topic of the document?",
    "PDF — Explain retrieval-augmented generation (RAG).",
    "PDF — In which year was chain-of-thought prompting demonstrated?",
    "PDF — What does the document say about system prompts?",
    "PDF — Summarize the section about transformers.",
    "PDF — What does the document say about RLHF?",
    "PDF — What are the key limitations discussed in the document?",
    "PDF — What examples of extensibility techniques are mentioned?",
    "PDF — Does the document mention GPT-4o? If yes, what does it say?",
    "PDF — Who is Shah Rukh Khan? (should say not in the document)",
]


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def _render_message(msg: dict):
    role = msg["role"]
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(msg.get("content", ""))

            meta = msg.get("meta") or {}
            route = meta.get("route")
            route_reason = meta.get("route_reason")

            cols = st.columns(2)
            with cols[0]:
                st.caption(f"Route: **{route or 'unknown'}**")
            with cols[1]:
                if route_reason:
                    st.caption(f"Reason: `{route_reason}`")

            citations = meta.get("citations") or []
            if route == "pdf" and citations:
                with st.expander("Citations"):
                    for c in citations:
                        st.write(f"- page={c.get('page')} | chunk_ref={c.get('chunk_ref')}")
        else:
            st.markdown(msg.get("content", ""))


_init_state()

with st.sidebar:
    st.subheader("Quick questions")
    st.caption("One-click prompts for reviewers (the router still decides the path).")

    q = st.selectbox("Pick a prompt", QUICK_QUESTIONS)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Ask", use_container_width=True):
            # Strip the label prefix before sending to the agent
            st.session_state.pending_query = q.split("—", 1)[-1].strip()
            st.rerun()
    with col_b:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()

for m in st.session_state.messages:
    _render_message(m)

user_query = st.chat_input("Ask about weather or ask from the PDF…")
if st.session_state.pending_query and not user_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    _render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent(user_query)

        answer = result.get("answer") or "Sorry—no answer was generated."
        st.markdown(answer)

        cols = st.columns(2)
        with cols[0]:
            st.caption(f"Route: **{result.get('route') or 'unknown'}**")
        with cols[1]:
            if result.get("route_reason"):
                st.caption(f"Reason: `{result.get('route_reason')}`")

        citations = result.get("citations") or []
        if result.get("route") == "pdf" and citations:
            with st.expander("Citations"):
                for c in citations:
                    st.write(f"- page={c.get('page')} | chunk_ref={c.get('chunk_ref')}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "meta": {
                "route": result.get("route"),
                "route_reason": result.get("route_reason"),
                "citations": result.get("citations") or [],
            },
        }
    )

