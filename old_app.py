import os
import tempfile
from typing import List

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_community.llms import Ollama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


# ---------------------------
# Load Documents
# ---------------------------
def load_documents(uploaded_files) -> List:
    docs = []

    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Choose loader
        if ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif ext == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(temp_path)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue

        file_docs = loader.load()

        for d in file_docs:
            d.metadata["source"] = uploaded_file.name

        docs.extend(file_docs)
        os.remove(temp_path)

    return docs


# ---------------------------
# Build Vectorstore
# ---------------------------
def build_vector_db(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
    )

    return vectorstore


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Knowledge Base Agent — Phi-3 + Ollama", layout="wide")

st.title("📚 AI Knowledge Base Agent — Local & Free (Phi-3 + Ollama)")
st.write("Upload documents and ask questions. 100% local. No API keys needed.")


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# Sidebar
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    build_btn = st.button("Build Knowledge Base")

    st.markdown("---")
    st.header("⚙ Model Settings")
    temperature = st.slider("Model Creativity", 0.0, 1.0, 0.3)


# ---------------------------
# Build Knowledge Base
# ---------------------------
if build_btn:
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Processing documents..."):

            docs = load_documents(uploaded_files)

            if len(docs) == 0:
                st.error("Could not load any documents.")
            else:
                vectorstore = build_vector_db(docs)
                st.session_state.vectorstore = vectorstore

                # Use Ollama + Phi-3
                llm = Ollama(
                    model="phi3",
                    temperature=temperature
                )

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                )

                st.session_state.qa_chain = qa

        st.success("🎉 Knowledge Base Built Successfully!")


# ---------------------------
# Chat Section
# ---------------------------
st.subheader("💬 Ask a Question")

if st.session_state.vectorstore is None:
    st.info("Upload documents and build the Knowledge Base first.")
else:
    query = st.text_input("Type your question here...")

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain({"query": query})

                answer = result["result"]
                sources = result["source_documents"]

                st.markdown("### 🧠 Answer")
                st.write(answer)

                st.markdown("### 📎 Sources")
                for i, doc in enumerate(sources, 1):
                    st.write(f"{i}. **{doc.metadata.get('source', 'Unknown')}**")
                    st.caption(doc.page_content[:200] + "...")
