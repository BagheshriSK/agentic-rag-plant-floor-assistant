import uuid
import subprocess
import tempfile
import os

import streamlit as st
from rag_pipeline.graph import pipeline

st.set_page_config(page_title="Plant Floor Assistant", layout="wide")

# Session state init
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

st.title("Plant Floor Documentation Assistant")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about safety, maintenance, or quality control...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    result = pipeline.invoke(
        {"raw_query": user_input},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )
    final_response = result.get("final_response") or {}
    st.session_state.last_response = final_response

    if "error" in final_response:
        answer_text = f"Error: {final_response['error']}"
    else:
        answer_text = final_response.get("answer", "")

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    st.rerun()

# Show result panels after last response
if st.session_state.last_response:
    resp = st.session_state.last_response

    if "error" in resp:
        st.error(resp["error"])
    else:
        reasoning = resp.get("reasoning_steps", "")
        if reasoning:
            with st.expander("🧠 Reasoning Steps"):
                st.markdown(reasoning)

        sources = resp.get("sources", [])
        if sources:
            with st.expander("📚 Sources"):
                st.dataframe(
                    [
                        {
                            "collection": s.get("collection", ""),
                            "document": s.get("document", ""),
                            "section": s.get("section", ""),
                            "chunk_id": s.get("chunk_id", ""),
                        }
                        for s in sources
                    ]
                )

        eval_scores = resp.get("eval_scores")
        if eval_scores:
            with st.expander("📊 Evaluation Scores"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Answer Relevance", f"{eval_scores.get('answer_relevance', 0):.2f}")
                col2.metric("Context Relevance", f"{eval_scores.get('context_relevance', 0):.2f}")
                col3.metric("Groundedness", f"{eval_scores.get('groundedness', 0):.2f}")
                col4.metric("Completeness", f"{eval_scores.get('completeness', 0):.2f}")

# Sidebar
with st.sidebar:
    st.header("📁 Upload to Knowledge Base")

    collection = st.selectbox(
        "Target Knowledge Base",
        ["safety_procedures", "maintenance_manuals", "quality_control_standards"],
    )

    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Upload & Ingest") and uploaded_files:
        tmp_dir = tempfile.mkdtemp()
        for f in uploaded_files:
            with open(os.path.join(tmp_dir, f.name), "wb") as out:
                out.write(f.read())
        result = subprocess.run(
            ["python", "scripts/ingest_pdfs.py", "--source-dir", tmp_dir, "--collection", collection],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            st.success(f"Ingested {len(uploaded_files)} file(s) into '{collection}'")
        else:
            st.error(f"Ingestion failed: {result.stderr}")

    st.divider()

    if st.button("🔄 New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_response = None
        st.rerun()
