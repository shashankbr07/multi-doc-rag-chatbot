# 🧠 DocMind AI — Multi-Document RAG Chatbot

> **Resume Project** | Gen AI · RAG · Google Gemini · ChromaDB · Streamlit

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload multiple PDFs or text files and have intelligent, cited conversations with them — powered by **Google Gemini** and **ChromaDB**.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📂 Multi-document upload | PDF + TXT files; unlimited documents |
| 🔍 Smart embedding detection | Auto-discovers the best available Gemini embedding model |
| 🔎 Hybrid OCR | Text extraction via pypdf; falls back to **Gemini Vision** for scanned pages |
| 💬 Streaming responses | Real-time typewriter-style output with full markdown rendering |
| 📌 Source citations | Answers cite which document + chunk they came from |
| 🔎 Context preview | Expandable retrieved chunk viewer |
| ⬇️ Chat export | Download full conversation as Markdown |
| 🎨 Premium dark UI | Glassmorphism design, smooth animations |
| 🗑️ Auto-sync cleanup | Removing files from uploader auto-clears them from vector store |

---

## 🏗️ Architecture

```
User Uploads PDFs/TXTs
        │
        ▼
  Text Extraction (pypdf)
        │
        ▼
  Scanned page? ──Yes──► Gemini Vision OCR
        │                      │
        No                     │
        │◄─────────────────────┘
        ▼
  Chunking (800 chars, 100 overlap)
        │
        ▼
  Gemini Embeddings (auto-detected model)
        │
        ▼
  ChromaDB Vector Store (in-memory)
        │
    [Query]
        │
        ▼
  Query Embedding → Top-K Retrieval
        │
        ▼
  Gemini 2.5 Flash → Streamed Answer + Citations
        │
        ▼
  Streamlit Chat UI — Markdown + Sources
```

---

## 🚀 Quick Start (Local)

### 1. Clone this project
```bash
git clone https://github.com/shashankbr07/multi-doc-rag-chatbot.git
cd multi-doc-rag-chatbot
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key
```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
```
Or just paste the key directly in the app sidebar at runtime.

Get a free API key at: https://aistudio.google.com/app/apikey

### 5. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this project to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo and set `app.py` as the main file
4. In **Advanced Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   ```
5. Click **Deploy** — live in ~2 minutes!

---

## 📁 Project Structure

```
multi-doc-rag-chatbot/
├── app.py                  # Streamlit UI (chat, sidebar, streaming)
├── rag_engine.py           # Core RAG pipeline (OCR, embedding, retrieval, generation)
├── requirements.txt
├── runtime.txt             # Python version for Streamlit Cloud
├── .env.example
├── .streamlit/
│   └── config.toml         # Dark theme configuration
├── README.md
└── sample_docs/
    ├── nexus_bank_annual_report_2024.txt
    └── nexus_bank_ai_strategy_2025.txt
```

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | Auto-detected (gemini-embedding-001 / text-embedding-004) |
| **Vector Store** | ChromaDB (in-memory) |
| **PDF Parsing** | pypdf + Gemini Vision OCR (hybrid) |
| **UI** | Streamlit (native chat components) |
| **Language** | Python 3.12 |

---

## 💡 Try These Sample Questions

After uploading the sample docs from `sample_docs/`:

1. *"What was Nexus Bank's net profit in FY2024?"*
2. *"Which AI models are currently deployed in production?"*
3. *"What is the capital adequacy ratio and how does it compare to the regulatory minimum?"*
4. *"Tell me about the GenAI customer support agent — how many queries did it handle?"*
5. *"What is the technology budget allocation for FY2025?"*

---

## 📄 License

MIT License — free to use, modify, and deploy.
