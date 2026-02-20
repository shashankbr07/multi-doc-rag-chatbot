# 🧠 DocMind AI — Multi-Document RAG Chatbot

> **Resume Project** | Gen AI · RAG · Google Gemini · ChromaDB · Streamlit

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload multiple PDFs or text files and have intelligent, cited conversations with them — powered by **Google Gemini** and **ChromaDB**.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📂 Multi-document upload | PDF + TXT files; unlimited documents |
| 🔍 Semantic search | Gemini `text-embedding-004` + ChromaDB vector store |
| 💬 Conversational AI | Gemini 2.0 Flash with 6-turn memory |
| 📌 Source citations | Answers cite which document they came from |
| 🔎 Context preview | Expandable retrieved chunk viewer |
| ⬇️ Chat export | Download full conversation as Markdown |
| 🎨 Premium dark UI | Glassmorphism design, smooth animations |

---

## 🏗️ Architecture

```
User Uploads PDFs/TXTs
        │
        ▼
  Text Extraction (pypdf)
        │
        ▼
  Chunking (800 chars, 100 overlap)
        │
        ▼
  Gemini Embeddings (text-embedding-004)
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
  Gemini 2.0 Flash → Answer + Citations
        │
        ▼
  Streamlit UI — Chat + Sources
```

---

## 🚀 Quick Start (Local)

### 1. Clone / download this project
```bash
cd /path/to/multi-doc-rag-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
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
├── app.py              # Streamlit UI
├── rag_engine.py       # Core RAG pipeline (embedding, retrieval, generation)
├── requirements.txt
├── .env.example
├── README.md
└── sample_docs/
    ├── nexus_bank_annual_report_2024.txt   # Demo: bank annual report
    └── nexus_bank_ai_strategy_2025.txt     # Demo: AI strategy whitepaper
```

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | Google text-embedding-004 |
| **Vector Store** | ChromaDB (in-memory) |
| **PDF Parsing** | pypdf |
| **UI** | Streamlit |
| **Language** | Python 3.10+ |

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
