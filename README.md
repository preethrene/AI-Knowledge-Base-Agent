📘 AI Knowledge Base Agent
Upload files → Ask questions → Local AI answers using Phi-3 + LangChain + ChromaDB + Streamlit










🚀 Overview

AI Knowledge Base Agent is a fully local, offline document Q&A system.

Upload PDF, TXT, DOCX files — the AI reads, indexes, and answers your questions using:

Microsoft Phi-3 (via Ollama)

LangChain RetrievalQA

ChromaDB vector store

SentenceTransformer embeddings

Streamlit UI

This system works 100% offline, is free, and runs entirely on your laptop.

📸 Screenshots
Home Page	Ask a Question

	

(Update image names to match your screenshot files)

🧠 Architecture Diagram
flowchart TD
    U[👤 User] --> UI[🖥️ Streamlit UI]
    UI --> VS[📂 ChromaDB Vector Store]
    UI --> LLM[🤖 Phi-3 via Ollama]
    VS --> LLM
    LLM --> UI

🛠 Features

📄 Upload multiple documents

⚙️ Automatic text extraction + chunking

🔍 Semantic search (vector retrieval)

🤖 Local LLM answers (Phi-3 via Ollama)

💬 Clean chat-style interface

💾 Optional chat history

📥 Export chat as PDF

🎨 Dark/Light mode

⌨️ Smooth AI typing animation

🔒 Fully offline and private

📦 Installation
1️⃣ Clone the repository
git clone https://github.com/preethrene/AI_Knowledge_Base_Agent.git
cd AI_Knowledge_Base_Agent

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install and pull Phi-3 model
ollama pull phi3

4️⃣ Run the app
streamlit run app.py

📝 License

MIT License © Preetham N

✔️ README Ready

You can safely paste this into GitHub — everything will render correctly 🎉

If you want a premium, animated, pro-level README, just say:
➡️ "Give me the premium README"

You said:
"Give me the premium README" 
ChatGPT said:

🔥 PREMIUM README COMING UP — FOR MAXIMUM IMPACT ON RECRUITERS & GITHUB VIEWERS
Designed like top-tier open-source projects 💯💼

👉 Copy–paste directly into GitHub README.md

🚀 AI Knowledge Base Agent
Upload → Index → Ask → Answer — 100% Local, Private & Offline AI
<p align="center"> <img src="https://img.shields.io/badge/Author-preethrene-blue" /> <img src="https://img.shields.io/badge/Python-3.10-yellow" /> <img src="https://img.shields.io/badge/Framework-Streamlit-red" /> <img src="https://img.shields.io/badge/LLM-Phi3-%2300b300" /> <img src="https://img.shields.io/badge/License-MIT-green" /> </p> <p align="center"> <b>A powerful offline AI system that reads your documents and answers questions instantly — using Phi-3, LangChain & ChromaDB.</b> </p>
✨ Key Highlights
🔒 100% Local & Private

All processing happens on your machine — no cloud, no API costs, no data leaks.

⚡ Instant Answers from Your Documents

Upload PDFs, TXT, or DOCX → AI builds a knowledge base and answers anything you ask.

🧠 Powered by

Microsoft Phi-3 (via Ollama)

LangChain RetrievalQA

ChromaDB vector store

SentenceTransformer embeddings

Streamlit UI

📸 Screenshots
Home Page	Ask a Question

	

(Rename images as needed to match your filenames)

🧩 Architecture
flowchart TD
    U[👤 User] --> UI[🖥️ Streamlit UI]
    UI --> VS[📂 ChromaDB Vector Store]
    UI --> LLM[🤖 Phi-3 via Ollama]
    VS --> LLM
    LLM --> UI

🛠 Features
📄 Document Handling

Upload PDF, TXT, DOCX

Automatic text extraction

Smart chunking + metadata

🔍 Smart Retrieval

ChromaDB vector search

MiniLM Sentence Embeddings

Fast + accurate semantic matching

🤖 Local LLM (Phi-3)

No API keys required

No data leaves your laptop

Smooth typing animation

🎛 Complete User Interface

Clean chat layout

Beautiful dark/light themes

Previous Q&A browsing

Download entire chat as PDF

⚙️ Installation Guide
1️⃣ Clone Repository
git clone https://github.com/preethrene/AI_Knowledge_Base_Agent.git
cd AI_Knowledge_Base_Agent

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Install Local LLM (Phi-3)
ollama pull phi3

4️⃣ Run the App
streamlit run app.py

📂 Project Structure
AI_Knowledge_Base_Agent/
│── app.py                 # Main application
│── requirements.txt       # Dependencies
│── chat_history.json      # Local saved history
│── screenshots/           # Screenshots for README
│── chroma_db/             # Auto-generated vector DB (ignored)
└── README.md              # Documentation

📥 Export Chat as PDF

Download your full conversation history with one click — great for reports or HR/Policy summaries.

🎨 UI Themes

🌙 Dark Mode

☀️ Light Mode

One toggle. Seamless switching.

🧪 Sample Use Cases
👨‍💼 HR Document Assistant

Ask about company policies, onboarding rules, leave structure, dress code, etc.

📚 Study/Notes Organizer

Upload textbooks or PDF study material and ask questions.

🗂 Office Document Search

Retrieve info from meeting minutes, SOPs, manuals.

📝 License

MIT License © Preetham N
