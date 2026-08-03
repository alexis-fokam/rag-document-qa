"""
app.py
------
Interface Streamlit du projet RAG Document Q&A.
Permet d'uploader un PDF/CSV et de poser des questions dessus.
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

# Charge les variables d'environnement depuis .env (en local uniquement ;
# sur Streamlit Cloud, on utilise st.secrets à la place, voir plus bas)
load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="centered")

st.title("📄 RAG Document Q&A")
st.caption(
    "Uploade un PDF ou un CSV, pose tes questions, et obtiens des réponses "
    "sourcées basées uniquement sur le contenu du document."
)

# --- Récupération de la clé API ---
# En local : elle vient du fichier .env
# Sur Streamlit Cloud : elle vient de st.secrets (configuré dans les settings de l'app)
google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

if not google_api_key:
    st.error(
        "⚠️ Clé API Google manquante. Ajoute GOOGLE_API_KEY dans ton fichier .env "
        "(en local) ou dans les Secrets Streamlit Cloud (en production)."
    )
    st.stop()

# --- Initialisation de l'état de session ---
# On garde le vectorstore en mémoire tant que l'utilisateur ne change pas de fichier,
# pour éviter de re-calculer les embeddings à chaque question.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Barre latérale : upload et infos sur le document ---
with st.sidebar:
    st.header("📁 Document")
    uploaded_file = st.file_uploader("Choisis un fichier PDF ou CSV", type=["pdf", "csv"])

    if uploaded_file is not None and st.session_state.current_file != uploaded_file.name:
        with st.spinner("📚 Lecture et indexation du document en cours..."):
            documents = load_document(uploaded_file)
            chunks = split_documents(documents)
            st.session_state.vectorstore = build_vectorstore(chunks, google_api_key)
            st.session_state.current_file = uploaded_file.name
            st.session_state.chunk_count = len(chunks)
            st.session_state.chat_history = []  # reset de l'historique sur nouveau doc
        st.success(f"✅ {len(chunks)} morceaux de texte indexés.")

    if st.session_state.current_file:
        st.divider()
        st.caption("📄 Document actif")
        st.markdown(f"**{st.session_state.current_file}**")
        st.caption(f"🧩 {st.session_state.chunk_count} morceaux indexés")

        if st.session_state.chat_history and st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# --- Zone de conversation ---
if st.session_state.vectorstore is not None:
    for exchange in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(exchange["question"])
        with st.chat_message("assistant", avatar="📄"):
            st.write(exchange["answer"])
            with st.expander(f"📎 Sources ({len(exchange['sources'])})"):
                for source in exchange["sources"]:
                    st.caption(f"**Page {source['page']}** — {source['excerpt']}")

    question = st.chat_input("Pose ta question sur le document...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant", avatar="📄"):
            with st.spinner("🔍 Recherche dans le document et génération de la réponse..."):
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
    st.info("👈 Commence par uploader un document dans la barre latérale pour pouvoir poser des questions.")
