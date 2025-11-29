\# 📘 AI Knowledge Base Agent  

\### Upload files → Ask questions → Local AI answers using Phi-3 + LangChain + ChromaDB + Streamlit



!\[Author](https://img.shields.io/badge/Author-preethrene-blue)

!\[Built With](https://img.shields.io/badge/Built%20With-Python%203.10-yellow)

!\[Framework](https://img.shields.io/badge/Framework-Streamlit-red)

!\[Model](https://img.shields.io/badge/Model-Phi3-%2300b300)

!\[License](https://img.shields.io/badge/License-MIT-green)



---



\## 🚀 Overview



\*\*AI Knowledge Base Agent\*\* is a fully local document Q\&A system.  

Upload PDFs, TXT, or DOCX — the AI reads, indexes, and answers questions using:



\- \*\*Microsoft Phi-3 (via Ollama)\*\*

\- \*\*LangChain RetrievalQA\*\*

\- \*\*ChromaDB vector store\*\*

\- \*\*SentenceTransformer embeddings\*\*

\- \*\*Streamlit frontend\*\*



💡 \*Everything runs 100% locally — no API keys, no internet required.\*



---



\## 🌟 Features



\### 🔍 Document Understanding  

\- Upload \*\*PDF / TXT / DOCX\*\*  

\- Extracts and chunks text  

\- Creates embeddings  

\- Stores vectors in \*\*ChromaDB\*\*



\### 🤖 Smart AI Q\&A  

\- Local LLM (\*\*Phi-3\*\*)  

\- Answers based on your files only  

\- Cites sources  

\- Clean chat UI with bubbles  

\- Typing animation for AI responses



\### 💾 Chat History  

\- Stored locally in `chat\_history.json`  

\- Last question never appears in the input box  

\- Can export entire chat as \*\*PDF\*\*



\### 🎨 Modern UI  

\- Light/Dark mode  

\- Professional header  

\- Gradient banners  

\- Clean layout  

\- Responsive design



---



\## 🏗 Architecture (High-Level)



```text

User Question

&nbsp;     │

&nbsp;     ▼

Streamlit UI

&nbsp;     │

&nbsp;     ▼

LangChain RetrievalQA

&nbsp;     │

&nbsp;     ├─> ChromaDB (similar chunks from documents)

&nbsp;     │

&nbsp;     └─> Phi-3 LLM (Ollama)

&nbsp;             │

&nbsp;             ▼

&nbsp;       Final Answer + Sources



