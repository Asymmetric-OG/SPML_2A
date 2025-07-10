📚 RAG Q&A with LangChain & Streamlit
Ask questions based on research papers using a simple Retrieval-Augmented Generation (RAG) system.

🚀 Features
Loads PDFs from the research/ folder

Splits and embeds text using all-mpnet-base-v2

Stores embeddings in a FAISS vector store

Uses flan-t5-base to generate answers

Streamlit-based interface for easy interaction

🧠 How It Works
PDFs are loaded and split into chunks

Chunks are embedded and stored in FAISS

Given a question, top-k relevant chunks are retrieved

flan-t5-base generates an answer using only the retrieved context

📁 File Overview
app.py: Streamlit frontend

rag_chain.py: RAG pipeline logic

research/: Folder with PDF documents

🛠 Built With
Hugging Face Transformers

LangChain

FAISS

Streamlit

