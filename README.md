# RAG Document Q&A

Application permettant d'uploader un document (PDF ou CSV) et de poser des
questions dessus en langage naturel. Les réponses sont générées **uniquement**
à partir du contenu du document (Retrieval-Augmented Generation), avec citation
des sources (pages).

## 🎯 Fonctionnalités

- Upload de PDF ou CSV
- Découpage automatique du texte en chunks
- Recherche sémantique (embeddings + base vectorielle Chroma)
- Génération de réponses sourcées via l'API Google Gemini
- Historique de conversation dans l'interface

## 🛠️ Stack technique

- **Python 3.11+**
- **LangChain** — orchestration du pipeline RAG
- **Chroma** — base de données vectorielle (en mémoire)
- **Google Gemini API** — embeddings + génération de texte
- **Streamlit** — interface utilisateur

## 🚀 Installation locale

```bash
git clone https://github.com/alexis-fokam/rag-document-qa.git
cd rag-document-qa
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Ouvre .env et colle ta clé Google API (gratuite sur https://aistudio.google.com/apikey)
streamlit run app.py
```

## 🌐 Démo en ligne

👉 (https://rag-document-appgit-xakwfynacexfzvzcnpkbex.streamlit.app/)

## 📂 Structure du projet

```
rag-document-qa/
├── app.py                      # Interface Streamlit
├── rag_pipeline.py             # Logique RAG (chargement, découpage, embeddings, génération)
├── requirements.txt            # Dépendances Python
├── .env.example                # Modèle pour la clé API
└── .streamlit/
    └── secrets.toml.example    # Modèle pour le déploiement cloud
```

## 👤 Auteur

Alexis Mvondo Fokam — [Portfolio](https://portfolio-alexis-fokam.netlify.app) — [LinkedIn](https://linkedin.com/in/alexis-fokam)
