"""
rag_pipeline.py
----------------
Ce fichier contient toute la logique "cerveau" du projet RAG :
1. Charger un document (PDF ou CSV)
2. Le découper en petits morceaux (chunks)
3. Transformer ces morceaux en vecteurs (embeddings) et les stocker
4. Récupérer les morceaux pertinents pour une question donnée
5. Générer une réponse avec le LLM, basée sur ces morceaux

Ce fichier ne contient AUCUN code d'interface (Streamlit) : il est
volontairement séparé pour rester réutilisable (API, CLI, autre UI...).
"""

import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document


def load_document(uploaded_file) -> List[Document]:
    """
    Charge un fichier uploadé (PDF ou CSV) et retourne une liste de
    Document LangChain (texte + métadonnées, ex: numéro de page).

    uploaded_file : objet retourné par st.file_uploader (Streamlit)
    """
    # On écrit temporairement le fichier sur disque, car les loaders
    # LangChain attendent un chemin de fichier, pas un objet en mémoire.
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if suffix.lower() == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix.lower() == ".csv":
        loader = CSVLoader(tmp_path)
    else:
        raise ValueError(f"Format non supporté : {suffix}. Utilise un PDF ou un CSV.")

    documents = loader.load()
    os.remove(tmp_path)  # Nettoyage du fichier temporaire
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Découpe les documents en petits morceaux (chunks) de ~1000 caractères,
    avec un chevauchement (overlap) de 150 caractères pour ne pas couper
    une idée en plein milieu.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks: List[Document], google_api_key: str) -> Chroma:
    """
    Transforme chaque chunk en vecteur (embedding) et les stocke dans
    une base vectorielle Chroma, gardée en mémoire (pas de fichier sauvegardé).
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_qa_session",
    )
    return vectorstore


def get_relevant_chunks(vectorstore: Chroma, question: str, k: int = 4) -> List[Document]:
    """
    Recherche les k morceaux de texte les plus pertinents pour la question.
    """
    return vectorstore.similarity_search(question, k=k)


def generate_answer(question: str, relevant_chunks: List[Document], google_api_key: str) -> dict:
    """
    Construit le prompt final (question + contexte récupéré) et appelle
    le LLM pour générer une réponse sourcée.

    Retourne un dict avec la réponse et les sources utilisées.
    """
    context_text = "\n\n---\n\n".join(
        f"[Source : page {doc.metadata.get('page', doc.metadata.get('row', '?'))}]\n{doc.page_content}"
        for doc in relevant_chunks
    )

    prompt = f"""Tu es un assistant qui répond UNIQUEMENT à partir du contexte fourni ci-dessous.
Si la réponse ne se trouve pas dans le contexte, dis clairement "Je ne trouve pas cette information dans le document."
Ne jamais inventer d'information qui n'est pas dans le contexte.

CONTEXTE :
{context_text}

QUESTION : {question}

RÉPONSE (en français, claire et concise) :"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.2,
    )
    response = llm.invoke(prompt)

    sources = [
        doc.metadata.get("page", doc.metadata.get("row", "?"))
        for doc in relevant_chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
