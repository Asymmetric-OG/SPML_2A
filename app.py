import streamlit as st
from rag_chain import answer_question, setup_vector_store

st.set_page_config(page_title="📚 RAG Q&A", layout="centered")
st.title("📄 Ask a Research Question")
st.write("Ask a question based on the uploaded research papers.")

@st.cache_resource
def initialize():
    setup_vector_store()
initialize()

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = answer_question(question)
        st.markdown("### 🧠 Answer")
        st.success(answer)
