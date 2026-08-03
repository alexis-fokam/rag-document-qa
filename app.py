"""
app.py
------
Streamlit interface for the RAG Document Q&A project.
Lets you upload a PDF/CSV and ask questions about it.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    load_document,
    split_documents,
    build_vectorstore,
    get_relevant_chunks,
    generate_answer,
)

# Load environment variables from .env (local use only;
# on Streamlit Cloud, st.secrets is used instead, see below)
load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="centered")

st.title("📄 RAG Document Q&A")
st.caption(
    "Upload a PDF or CSV, ask your questions, and get answers "
    "sourced exclusively from the document's content."
)

# --- API key retrieval ---
# Locally: it comes from the .env file
# On Streamlit Cloud: it comes from st.secrets (configured in the app settings)
google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

if not google_api_key:
    st.error(
        "⚠️ Missing Google API key. Add GOOGLE_API_KEY to your .env file "
        "(locally) or to the Streamlit Cloud Secrets (in production)."
    )
    st.stop()

# --- Session state initialization ---
# We keep the vectorstore in memory as long as the user doesn't change files,
# to avoid recomputing embeddings on every question.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: upload and document info ---
with st.sidebar:
    st.header("📁 Document")
    uploaded_file = st.file_uploader("Choose a PDF or CSV file", type=["pdf", "csv"])

    if uploaded_file is not None and st.session_state.current_file != uploaded_file.name:
        with st.spinner("📚 Reading and indexing the document..."):
            documents = load_document(uploaded_file)
            chunks = split_documents(documents)
            st.session_state.vectorstore = build_vectorstore(chunks, google_api_key)
            st.session_state.current_file = uploaded_file.name
            st.session_state.chunk_count = len(chunks)
            st.session_state.chat_history = []  # reset history on new document
        st.success(f"✅ {len(chunks)} text chunks indexed.")

    if st.session_state.current_file:
        st.divider()
        st.caption("📄 Active document")
        st.markdown(f"**{st.session_state.current_file}**")
        st.caption(f"🧩 {st.session_state.chunk_count} chunks indexed")

        if st.session_state.chat_history and st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# --- Chat area ---
if st.session_state.vectorstore is not None:
    for exchange in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(exchange["question"])
        with st.chat_message("assistant", avatar="📄"):
            st.write(exchange["answer"])
            with st.expander(f"📎 Sources ({len(exchange['sources'])})"):
                for source in exchange["sources"]:
                    st.caption(f"**Page {source['page']}** — {source['excerpt']}")

    question = st.chat_input("Ask a question about the document...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant", avatar="📄"):
            with st.spinner("🔍 Searching the document and generating the answer..."):
                relevant_chunks = get_relevant_chunks(st.session_state.vectorstore, question)
                result = generate_answer(question, relevant_chunks, google_api_key)
            st.write(result["answer"])
            with st.expander(f"📎 Sources ({len(result['sources'])})"):
                for source in result["sources"]:
                    st.caption(f"**Page {source['page']}** — {source['excerpt']}")

        st.session_state.chat_history.append(
            {"question": question, "answer": result["answer"], "sources": result["sources"]}
        )
else:
    st.info("👈 Start by uploading a document in the sidebar to ask questions.")
