# 📘 AI Knowledge Base Agent  

### Upload files → Ask questions → Local AI answers using Phi-3 + LangChain + ChromaDB + Streamlit  

![Author](https://img.shields.io/badge/Author-preethrene-blue)
![Built With](https://img.shields.io/badge/Built%20With-Python%203.10-yellow)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Model](https://img.shields.io/badge/Model-Phi3-%2300b300)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Overview

**AI Knowledge Base Agent** is a fully local, offline document Q&A system.

Upload **PDF, TXT, DOCX** files — the AI reads, indexes, and answers your questions using:

- **Microsoft Phi-3 (via Ollama)**
- **LangChain RetrievalQA**
- **ChromaDB vector store**
- **SentenceTransformer embeddings**
- **Streamlit UI**

This system works **100% offline**, is **free**, and runs entirely on your laptop.

---

## 📸 Screenshots

| Home Page | Ask a Question |
|----------|----------------|
| ![Home](screenshots/screen1.png) | ![Ask](screenshots/screen2.png) |

*(Update filenames according to your screenshot names)*

---

## 🧠 Architecture Diagram

```mermaid
flowchart TD
    U[User] --> UI[Streamlit UI]
    UI --> VS[ChromaDB Vector Store]
    UI --> LLM[Phi-3 via Ollama]
    VS --> LLM
    LLM --> UI

🛠 Features

📄 Upload multiple documents
⚙️ Auto-indexing
🔍 Intelligent search (semantic retrieval)
🤖 Local LLM answers (Phi-3)
💾 Optional chat history
📥 Export chat as PDF
🎨 Dark/Light mode
⚡ Smooth typing animation

📦 Installation
git clone https://github.com/preethrene/AI_Knowledge_Base_Agent.git
cd AI_Knowledge_Base_Agent
pip install -r requirements.txt

Download the Phi-3 model:
ollama pull phi3

Run the app:
streamlit run app.py

📝 License

MIT License © Preetham N

---

# ✔️ Your README will now look perfect  
Badges will show correctly, sections are clean, and screenshots will load beautifully.

---

# Want me to completely rewrite the README in a **premium professional style** (like top GitHub projects)?  
### → I can make it 10× more impressive for recruiters.
