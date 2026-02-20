"""
Multi-Document RAG Chatbot — Streamlit App
Upload multiple PDFs/TXTs, chat with them, and see cited sources.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI — Multi-Doc RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}

/* ── Chat messages ── */
.chat-user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(102,126,234,0.35);
    line-height: 1.6;
}

.chat-assistant {
    background: rgba(255,255,255,0.07);
    color: #e8eaf6;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 85%;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(8px);
    line-height: 1.6;
}

/* ── Source pill ── */
.source-pill {
    display: inline-block;
    background: rgba(102,126,234,0.2);
    border: 1px solid rgba(102,126,234,0.4);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78em;
    margin: 3px 4px 3px 0;
    white-space: nowrap;
}

/* ── Doc card ── */
.doc-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    color: #e0e0e0;
}
.doc-card .doc-name { font-weight: 600; color: #a5b4fc; font-size: 0.9em; }
.doc-card .doc-meta { font-size: 0.78em; color: #94a3b8; margin-top: 4px; }

/* ── Metrics ── */
.metric-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-box .metric-val { font-size: 1.8em; font-weight: 700; color: #a5b4fc; }
.metric-box .metric-lbl { font-size: 0.78em; color: #94a3b8; margin-top: 2px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102,126,234,0.45);
}

/* ── Input box ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e8eaf6 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Scrollable chat area ── */
.chat-container {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 6px;
    scroll-behavior: smooth;
}

/* ── Header ── */
.header-title {
    font-size: 2em;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc, #c4b5fd, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-sub {
    color: #94a3b8;
    font-size: 0.95em;
    margin-top: -8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "rag":           None,
        "chat_history":  [],
        "doc_stats":     [],
        "api_key_valid": False,
        "thinking":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind AI")
    st.markdown("<small style='color:#94a3b8'>Multi-Document RAG Chatbot</small>", unsafe_allow_html=True)
    st.divider()

    # ── API Key ──
    st.markdown("### 🔑 Gemini API Key")
    env_key     = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Enter your key",
        value=env_key,
        type="password",
        placeholder="AIza...",
        label_visibility="collapsed",
    )
    if api_key_input:
        _CODE_VERSION = 4  # bump this to force re-init after code changes
        # Recreate RAGEngine if key changed, not initialized, or code version changed
        needs_init = (
            st.session_state.rag is None
            or not st.session_state.api_key_valid
            or st.session_state.get("_last_key") != api_key_input
            or st.session_state.get("_code_ver") != _CODE_VERSION
        )
        if needs_init:
            try:
                st.session_state.rag           = RAGEngine(api_key=api_key_input)
                st.session_state.api_key_valid = True
                st.session_state._last_key     = api_key_input
                st.session_state._code_ver     = _CODE_VERSION
                st.session_state.doc_stats     = []
                st.session_state.chat_history  = []
                st.success(f"✅ API key accepted! Embed model: `{st.session_state.rag.embed_model}`", icon="🔓")
            except Exception as e:
                st.error(f"Invalid key: {e}")
                st.session_state.api_key_valid = False
    else:
        st.info("Add your [Gemini API key](https://aistudio.google.com/app/apikey) to get started.", icon="ℹ️")

    st.divider()

    # ── Document Upload ──
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.session_state.api_key_valid:
        # Auto-sync: remove docs from ChromaDB that were removed from uploader
        uploaded_names = {f.name for f in uploaded_files}
        removed_docs = [d for d in st.session_state.doc_stats if d["file_name"] not in uploaded_names]
        for doc in removed_docs:
            try:
                st.session_state.rag.remove_document(doc["file_name"])
            except Exception:
                pass
        if removed_docs:
            st.session_state.doc_stats = [d for d in st.session_state.doc_stats if d["file_name"] in uploaded_names]
            st.info(f"🗑️ Removed {len(removed_docs)} document(s) from index.", icon="🔄")

        # Index new files
        existing_names = {d["file_name"] for d in st.session_state.doc_stats}
        new_files      = [f for f in uploaded_files if f.name not in existing_names]
        if new_files:
            for f in new_files:
                with st.spinner(f"Indexing **{f.name}**…"):
                    try:
                        stats = st.session_state.rag.ingest_document(f.name, f.read())
                        st.session_state.doc_stats.append(stats)
                        ocr_info = f" (🔍 {stats['ocr_pages']} OCR pages)" if stats.get('ocr_pages', 0) > 0 else ""
                        st.success(f"✅ {f.name} indexed ({stats['chunks']} chunks){ocr_info}", icon="📄")
                    except Exception as e:
                        st.error(f"Error processing {f.name}: {e}")

    elif uploaded_files and not st.session_state.api_key_valid:
        st.warning("⚠️ Please enter a valid API key first.", icon="🔑")

    st.divider()

    # ── Document Library ──
    st.markdown("### 📚 Document Library")
    if st.session_state.doc_stats:
        for doc in st.session_state.doc_stats:
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-name">📄 {doc['file_name']}</div>
                <div class="doc-meta">
                    {doc['pages']} page(s) &nbsp;·&nbsp; {doc['chunks']} chunks &nbsp;·&nbsp;
                    {doc['chars']:,} chars{' &nbsp;·&nbsp; 🔍 ' + str(doc.get('ocr_pages', 0)) + ' OCR' if doc.get('ocr_pages', 0) > 0 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear All Documents", use_container_width=True):
            for doc in st.session_state.doc_stats:
                try:
                    st.session_state.rag.remove_document(doc["file_name"])
                except Exception:
                    pass
            st.session_state.doc_stats    = []
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown("<small style='color:#64748b'>No documents uploaded yet.</small>", unsafe_allow_html=True)

    st.divider()

    # ── Stats ──
    n_docs   = len(st.session_state.doc_stats)
    n_chunks = st.session_state.rag.total_chunks() if st.session_state.rag else 0
    n_turns  = len([m for m in st.session_state.chat_history if m["role"] == "user"])

    col1, col2, col3 = st.columns(3)
    for col, val, lbl in zip(
        [col1, col2, col3],
        [n_docs, n_chunks, n_turns],
        ["Docs", "Chunks", "Turns"],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Main Panel
# ─────────────────────────────────────────────
st.markdown("""
<div>
    <div class="header-title">🧠 DocMind AI</div>
    <div class="header-sub">Chat with your documents — powered by Google Gemini + RAG</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Welcome card ──
if not st.session_state.chat_history:
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(165,180,252,0.2);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        color: #cbd5e1;
        line-height: 1.7;
    ">
        <h3 style="color:#a5b4fc; margin-top:0;">👋 Welcome to DocMind AI</h3>
        <p>This is a <strong>Retrieval-Augmented Generation (RAG)</strong> chatbot that lets you have intelligent conversations with your own documents.</p>
        <b>How to get started:</b>
        <ol>
            <li>🔑 Paste your <a href="https://aistudio.google.com/app/apikey" style="color:#a5b4fc">Gemini API key</a> in the sidebar</li>
            <li>📂 Upload one or more <strong>PDF or TXT</strong> documents</li>
            <li>💬 Ask any question — the AI answers using your documents with <strong>cited sources</strong></li>
        </ol>
        <p style="margin-bottom:0; color:#64748b; font-size:0.85em;">
            Built with Google Gemini · ChromaDB · Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Chat history ──
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

        # Render source citations for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            unique_sources = {}
            for s in msg["sources"]:
                key = s["doc_name"]
                if key not in unique_sources:
                    unique_sources[key] = s

            pills_html = "".join(
                f'<span class="source-pill">📄 {s["doc_name"]}</span>'
                for s in unique_sources.values()
            )
            st.markdown(
                f'<div style="margin-top:4px; margin-bottom:10px;">'
                f'<small style="color:#64748b;">Sources: </small>{pills_html}</div>',
                unsafe_allow_html=True,
            )

            # Expandable chunk preview
            with st.expander("🔍 View retrieved context chunks"):
                for idx, s in enumerate(msg["sources"][:4]):
                    st.markdown(
                        f"**[Source {idx+1}] {s['doc_name']}** — chunk #{s['chunk_idx']}",
                    )
                    st.markdown(
                        f"> {s['chunk'][:400]}{'…' if len(s['chunk']) > 400 else ''}",
                    )
                    if idx < len(msg["sources"]) - 1:
                        st.divider()

# ── Chat input ──
user_input = st.chat_input(
    "Ask a question about your documents…",
    disabled=not st.session_state.api_key_valid or not st.session_state.doc_stats,
)

if user_input:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Stream assistant response with proper markdown
    with st.chat_message("assistant", avatar="🤖"):
        try:
            result = st.session_state.rag.query(
                question=user_input,
                top_k=6,
                chat_history=st.session_state.chat_history,
            )

            # st.write_stream renders markdown properly inside chat_message
            streamed_text = st.write_stream(result["stream"])

            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": streamed_text,
                "sources": result["sources"],
            })
        except Exception as e:
            error_msg = f"❌ Error: {e}"
            st.markdown(error_msg)
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": error_msg,
                "sources": [],
            })
    st.rerun()

# ── Footer actions ──
if st.session_state.chat_history:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col_b:
        # Build markdown export
        md_lines = ["# DocMind AI — Chat Export\n"]
        for m in st.session_state.chat_history:
            role = "**You**" if m["role"] == "user" else "**DocMind AI**"
            md_lines.append(f"{role}: {m['content']}\n")
            if m.get("sources"):
                src_names = list({s["doc_name"] for s in m["sources"]})
                md_lines.append(f"*Sources: {', '.join(src_names)}*\n")
            md_lines.append("---\n")
        st.download_button(
            label="⬇️ Export Chat (Markdown)",
            data="\n".join(md_lines),
            file_name="docmind_chat_export.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ── Auto-scroll JS ──
st.markdown("""
<script>
    const chatBox = document.getElementById('chat-box');
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
</script>
""", unsafe_allow_html=True)
