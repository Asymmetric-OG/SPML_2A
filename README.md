# 📚 RAG q&a with langchain & streamlit

ask questions based on research papers using a simple retrieval-augmented generation (rag) system.

---

## 🚀 features

- loads pdfs from the `research/` folder  
- splits and embeds text using `all-mpnet-base-v2`  
- stores embeddings in a faiss vector store  
- uses `flan-t5-base` to generate answers  
- streamlit-based interface for easy interaction

---

## 🧠 how it works

1. pdfs are loaded and split into chunks  
2. chunks are embedded and stored in faiss  
3. given a question, top-k relevant chunks are retrieved  
4. `flan-t5-base` generates an answer using only the retrieved context

---

## 📁 file overview

- `app.py`: streamlit frontend  
- `rag_chain.py`: rag pipeline logic  
- `research/`: folder with pdf documents

---

## 🛠 built with

- hugging face transformers  
- langchain  
- faiss  
- streamlit
