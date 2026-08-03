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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Upload du fichier ---
uploaded_file = st.file_uploader("Choisis un fichier PDF ou CSV", type=["pdf", "csv"])

if uploaded_file is not None:
    # Si c'est un nouveau fichier (différent du précédent), on reconstruit l'index
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("📚 Lecture et indexation du document en cours..."):
            documents = load_document(uploaded_file)
            chunks = split_documents(documents)
            st.session_state.vectorstore = build_vectorstore(chunks, google_api_key)
            st.session_state.current_file = uploaded_file.name
            st.session_state.chat_history = []  # reset de l'historique sur nouveau doc

        st.success(f"✅ Document indexé : {len(chunks)} morceaux de texte prêts à être interrogés.")

# --- Zone de question / réponse ---
if st.session_state.vectorstore is not None:
    st.divider()
    question = st.text_input("Pose ta question sur le document :")

    if st.button("Envoyer", type="primary") and question.strip():
        with st.spinner("🔍 Recherche dans le document et génération de la réponse..."):
            relevant_chunks = get_relevant_chunks(st.session_state.vectorstore, question)
            result = generate_answer(question, relevant_chunks, google_api_key)

        st.session_state.chat_history.append(
            {"question": question, "answer": result["answer"], "sources": result["sources"]}
        )

    # Affichage de l'historique (le plus récent en premier)
    for exchange in reversed(st.session_state.chat_history):
        st.markdown(f"**❓ {exchange['question']}**")
        st.write(exchange["answer"])
        st.caption(f"📎 Sources : {exchange['sources']}")
        st.divider()
else:
    st.info("👆 Commence par uploader un document pour pouvoir poser des questions.")
