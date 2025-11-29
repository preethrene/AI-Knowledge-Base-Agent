import sys
sys.modules["torch.classes"] = None  # Prevent Streamlit torch watcher crash

import os
import io
import json
import time
import tempfile
from typing import List

import streamlit as st

# ============== CONFIG ==============
st.set_page_config(page_title="AI Knowledge Base Agent", layout="wide")

HISTORY_FILE = "chat_history.json"

# ============== CSS THEMES ==============

DARK_THEME_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f19;
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}

/* Header gradient box */
.hero-box {
    background: linear-gradient(90deg, #38bdf8, #6366f1, #a855f7);
    padding: 24px;
    border-radius: 22px;
    margin-bottom: 20px;
    color: white;
    font-size: 26px;
    font-weight: 700;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}

/* Subtitle in header */
.hero-sub {
    font-size: 14px;
    font-weight: 400;
    color: #e5e7eb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827;
    color: white;
    border-right: 1px solid #1f2937;
}

/* File uploader box */
[data-testid="stFileUploader"] section {
    border-radius: 12px;
    border: 1px dashed #4b5563;
    background: #020617;
}

/* Text input (question box) */
.stTextInput > div > div > input {
    background: #1f2937 !important;
    color: #e5e7eb !important;
    border-radius: 10px !important;
    padding: 10px !important;
    border: 1px solid #374151 !important;
}

/* User bubble */
.chat-bubble-user {
    background: #1e3a8a;
    color: white;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: 8px;
}

/* AI bubble */
.chat-bubble-ai {
    background: #111827;
    color: #e5e7eb;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 85%;
    border: 1px solid #374151;
    margin-bottom: 8px;
}

/* Source card */
.source-card {
    background: #1f2937;
    border: 1px solid #374151;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 6px;
    color: #d1d5db;
}

/* Buttons */
.stButton > button {
    border-radius: 999px;
    font-weight: 600;
}
</style>
"""

LIGHT_THEME_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #f3f4f6;
    color: #111827;
    font-family: 'Inter', sans-serif;
}

/* Header box */
.hero-box {
    background: linear-gradient(90deg, #6366f1, #22c55e);
    padding: 24px;
    border-radius: 22px;
    margin-bottom: 20px;
    color: white;
    font-size: 26px;
    font-weight: 700;
    box-shadow: 0 8px 18px rgba(15,23,42,0.2);
}

.hero-sub {
    font-size: 14px;
    font-weight: 400;
    color: #e5e7eb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    color: #111827;
    border-right: 1px solid #e5e7eb;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    border-radius: 12px;
    border: 1px dashed #9ca3af;
    background: #f9fafb;
}

/* Question input */
.stTextInput > div > div > input {
    background: #ffffff !important;
    color: #111827 !important;
    border-radius: 10px !important;
    padding: 10px !important;
    border: 1px solid #d1d5db !important;
}

/* User bubble */
.chat-bubble-user {
    background: #2563eb;
    color: white;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: 8px;
}

/* AI bubble */
.chat-bubble-ai {
    background: #e5e7eb;
    color: #111827;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 85%;
    border: 1px solid #cbd5f5;
    margin-bottom: 8px;
}

/* Source card */
.source-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 6px;
    color: #374151;
}
</style>
"""

# ============== SIDEBAR (Theme + Upload + Build) ==============
with st.sidebar:
    st.title("⚙️ Controls")

    dark_mode = st.toggle("🌗 Dark mode", value=True)

    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    build_btn = st.button("🚀 Build Knowledge Base")

    st.markdown("---")
    st.markdown("Chat will be saved locally in `chat_history.json`.")

# Apply theme CSS
st.markdown(DARK_THEME_CSS if dark_mode else LIGHT_THEME_CSS, unsafe_allow_html=True)

# ============== HEADER ==============
st.markdown("""
<div class="hero-box">
📘 AI Knowledge Base Agent
<br>
<span class="hero-sub">
Upload files → Ask anything → 100% Local AI (Phi-3) · Offline · Free
</span>
</div>
""", unsafe_allow_html=True)

# ============== LANGCHAIN IMPORTS ==============
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ============== CHAT HISTORY HELPERS ==============

def load_history_from_file() -> List[dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history_to_file(history: List[dict]) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def generate_pdf_from_history(history: List[dict]) -> bytes:
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, "AI Knowledge Base Chat History")
    y -= 30
    p.setFont("Helvetica", 11)

    for item in history:
        q = "Q: " + item.get("q", "")
        a = "A: " + item.get("a", "")

        for line in [q, a, ""]:
            wrapped = []
            while len(line) > 90:
                wrapped.append(line[:90])
                line = line[90:]
            wrapped.append(line)

            for wline in wrapped:
                if y < 50:
                    p.showPage()
                    y = height - 50
                    p.setFont("Helvetica", 11)
                p.drawString(50, y, wline)
                y -= 14

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer.getvalue()

# ============== DOCUMENT LOADER ==============

def load_documents(uploaded_files, progress_bar=None) -> List:
    docs = []
    total = len(uploaded_files) if uploaded_files else 0

    for idx, uploaded_file in enumerate(uploaded_files):
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        if ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif ext == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(temp_path)
        else:
            os.remove(temp_path)
            continue

        file_docs = loader.load()
        for d in file_docs:
            d.metadata["source"] = uploaded_file.name
        docs.extend(file_docs)

        os.remove(temp_path)

        if progress_bar is not None and total > 0:
            progress_bar.progress((idx + 1) / total)

    return docs

# ============== VECTOR DB ==============

def build_vector_db(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
    )
    return vectorstore

# ============== SESSION STATE ==============
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "history" not in st.session_state:
    st.session_state.history = load_history_from_file()

# ============== BUILD KNOWLEDGE BASE ==============
if build_btn:
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Processing documents..."):
            progress = st.progress(0)
            docs = load_documents(uploaded_files, progress_bar=progress)
            progress.empty()

            if len(docs) == 0:
                st.error("No readable text extracted from the uploaded files.")
            else:
                vectorstore = build_vector_db(docs)
                st.session_state.vectorstore = vectorstore

                # ✅ Use phi3 (you have this installed)
                llm = Ollama(model="phi3")

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                )

                st.session_state.qa_chain = qa

        st.success("🎉 Knowledge Base Ready!")

# ============== CHAT SECTION ==============
st.subheader("💬 Ask a Question")

# Show previous chat history
if st.session_state.history:
    for msg in st.session_state.history:
        st.markdown(
            f"<div class='chat-bubble-user'>{msg['q']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='chat-bubble-ai'>{msg['a']}</div>",
            unsafe_allow_html=True,
        )

query = st.text_input(
    "",
    placeholder="Type your question here...",
    label_visibility="collapsed"
)

if st.session_state.vectorstore is None or st.session_state.qa_chain is None:
    st.info("📘 Upload documents and click **Build Knowledge Base** to start.")
else:
    if st.button("Ask"):
        if query.strip():
            # Show user bubble
            st.markdown(
                f"<div class='chat-bubble-user'>{query}</div>",
                unsafe_allow_html=True,
            )

            # Get answer
            with st.spinner("🤖 Thinking..."):
                result = st.session_state.qa_chain({"query": query})

            answer = result["result"]
            sources = result["source_documents"]

            # Typing animation for AI answer
            placeholder = st.empty()
            typed = ""
            for ch in answer:
                typed += ch
                placeholder.markdown(
                    f"<div class='chat-bubble-ai'>{typed}</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.004)

            # Save to history (session + JSON)
            st.session_state.history.append({"q": query, "a": answer})
            save_history_to_file(st.session_state.history)

            # Sources
            st.markdown("### 📎 Sources")
            for doc in sources:
                st.markdown(
                    f"<div class='source-card'><b>{doc.metadata.get('source')}</b><br>{doc.page_content[:200]}...</div>",
                    unsafe_allow_html=True,
                )

# ============== EXPORT CHAT AS PDF ==============
if st.session_state.history:
    pdf_bytes = generate_pdf_from_history(st.session_state.history)
    st.download_button(
        "📄 Download Chat as PDF",
        data=pdf_bytes,
        file_name="chat_history.pdf",
        mime="application/pdf",
    )
