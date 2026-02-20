"""
RAG Engine — Core logic for Multi-Document RAG Chatbot
Handles: PDF parsing, chunking, embedding (Gemini), vector storage (ChromaDB), retrieval & generation
Uses: google-genai SDK (new, replaces deprecated google-generativeai)
"""

from __future__ import annotations

import io
import os
import uuid
import time
from typing import Optional, List

import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]


def extract_text_from_pdf(file_bytes: bytes, gemini_client=None, chat_model: str = "gemini-2.5-flash"):
    """
    Hybrid PDF text extraction:
      1. Try pypdf text extraction per page
      2. For pages with <50 chars (likely scanned/image), use Gemini Vision OCR
    Returns (full_text, page_count, ocr_pages_count).
    """
    import base64

    reader = PdfReader(io.BytesIO(file_bytes))
    page_texts = []
    ocr_count = 0

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()

        # If text extraction returned very little, try Gemini Vision OCR
        if len(text) < 50 and gemini_client is not None:
            try:
                # Convert PDF page to image bytes via Gemini's native PDF support
                page_pdf = _extract_single_page_pdf(file_bytes, i)
                response = gemini_client.models.generate_content(
                    model=chat_model,
                    contents=[
                        types.Part.from_bytes(data=page_pdf, mime_type="application/pdf"),
                        "Extract ALL text from this document page. Return only the extracted text, nothing else.",
                    ],
                )
                ocr_text = response.text.strip()
                if len(ocr_text) > len(text):
                    text = ocr_text
                    ocr_count += 1
            except Exception:
                pass  # Fall back to whatever pypdf got

        page_texts.append(text)

    return "\n".join(page_texts), len(reader.pages), ocr_count


def _extract_single_page_pdf(full_pdf_bytes: bytes, page_index: int) -> bytes:
    """Extract a single page from a PDF as a new PDF byte stream."""
    from pypdf import PdfWriter
    reader = PdfReader(io.BytesIO(full_pdf_bytes))
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def extract_text_from_txt(file_bytes: bytes):
    """Return (full_text, page_count=1) from a plain-text byte stream."""
    return file_bytes.decode("utf-8", errors="replace"), 1


# --------------------------------------------------------------------------- #
#  RAG Engine
# --------------------------------------------------------------------------- #

class RAGEngine:
    """
    Encapsulates the full RAG pipeline:
      1. Ingest documents → chunk → embed → store in ChromaDB
      2. Query → retrieve top-k chunks → generate answer with Gemini
    """

    # Embedding models to try, in order of preference
    _EMBED_CANDIDATES = [
        "gemini-embedding-001",
        "text-embedding-004",
        "embedding-001",
        "models/gemini-embedding-001",
        "models/text-embedding-004",
        "models/embedding-001",
    ]
    CHAT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str):
        # Try both v1 and v1beta to find a working embed model
        self._client = None
        self.embed_model = None
        errors = []

        for api_ver in ["v1", "v1beta"]:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version=api_ver),
            )
            for model_name in self._EMBED_CANDIDATES:
                try:
                    client.models.embed_content(
                        model=model_name,
                        contents=["test"],
                    )
                    # If we get here, this model works!
                    self._client = client
                    self.embed_model = model_name
                    break
                except Exception as e:
                    errors.append(f"{api_ver}/{model_name}: {e}")
                    continue
            if self.embed_model:
                break

        if self._client is None or self.embed_model is None:
            # Show the first error to help diagnose
            first_err = errors[0] if errors else "No models tried"
            raise ValueError(
                f"No working embedding model found. First error: {first_err}"
            )

        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(
            name="rag_docs",
            metadata={"hnsw:space": "cosine"},
        )
        self.doc_registry: dict = {}  # file_name → stats dict

    # ------------------------------------------------------------------ #
    #  Ingestion
    # ------------------------------------------------------------------ #

    def ingest_document(self, file_name: str, file_bytes: bytes) -> dict:
        """Parse, chunk, embed and store a document. Returns stats dict."""
        ext = os.path.splitext(file_name)[1].lower()
        ocr_pages = 0
        if ext == ".pdf":
            text, pages, ocr_pages = extract_text_from_pdf(
                file_bytes,
                gemini_client=self._client,
                chat_model=self.CHAT_MODEL,
            )
        else:
            text, pages = extract_text_from_txt(file_bytes)

        chunks = _chunk_text(text)
        doc_id = str(uuid.uuid4())[:8]

        BATCH = 50
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i: i + BATCH]

            response = self._client.models.embed_content(
                model=self.embed_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            embeddings = [e.values for e in response.embeddings]

            self._collection.add(
                ids=[f"{doc_id}_{i + j}" for j in range(len(batch))],
                embeddings=embeddings,
                documents=batch,
                metadatas=[
                    {"doc_name": file_name, "doc_id": doc_id, "chunk_idx": i + j}
                    for j in range(len(batch))
                ],
            )
            if i + BATCH < len(chunks):
                time.sleep(0.2)

        stats = {
            "doc_id": doc_id,
            "file_name": file_name,
            "pages": pages,
            "chunks": len(chunks),
            "chars": len(text),
            "ocr_pages": ocr_pages,
        }
        self.doc_registry[file_name] = stats
        return stats

    def remove_document(self, file_name: str):
        """Remove all chunks for a document from the vector store."""
        if file_name not in self.doc_registry:
            return
        doc_id = self.doc_registry[file_name]["doc_id"]
        results = self._collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
        del self.doc_registry[file_name]

    # ------------------------------------------------------------------ #
    #  Retrieval & Generation
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        top_k: int = 6,
        chat_history: Optional[List[dict]] = None,
    ) -> dict:
        """
        Retrieve relevant chunks, then generate an answer with citations.
        Returns: {"answer": str, "sources": [...]}
        """
        if self._collection.count() == 0:
            return {
                "answer": "⚠️ No documents uploaded yet. Please upload at least one PDF or TXT file.",
                "sources": [],
            }

        # 1. Embed the query
        q_response = self._client.models.embed_content(
            model=self.embed_model,
            contents=[question],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        q_embedding = q_response.embeddings[0].values

        # 2. Retrieve top-k chunks
        results = self._collection.query(
            query_embeddings=[q_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks    = results["documents"][0]
        metadatas = results["metadatas"][0]

        # 3. Build context block
        context_parts = []
        for idx, (chunk, meta) in enumerate(zip(chunks, metadatas)):
            context_parts.append(
                f"[Source {idx+1} | Document: {meta['doc_name']} | Chunk #{meta['chunk_idx']}]\n{chunk}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 4. Build conversation history (last 6 turns)
        history_str = ""
        if chat_history:
            for turn in chat_history[-6:]:
                role = "User" if turn["role"] == "user" else "Assistant"
                history_str += f"{role}: {turn['content']}\n"

        # 5. Build prompt
        prompt = f"""You are a knowledgeable AI assistant. Answer the user's question using ONLY the context below.
For each key claim, cite the source like: [Source N].
If the answer is not in the context, say "I don't have enough information in the uploaded documents to answer that."

=== CONTEXT ===
{context}

=== CONVERSATION HISTORY ===
{history_str}

=== QUESTION ===
{question}

=== ANSWER (with citations) ==="""

        response = self._client.models.generate_content_stream(
            model=self.CHAT_MODEL,
            contents=prompt,
        )

        sources = [
            {"doc_name": meta["doc_name"], "chunk": chunk, "chunk_idx": meta["chunk_idx"]}
            for chunk, meta in zip(chunks, metadatas)
        ]

        def stream_with_sources():
            full_answer = []
            for chunk in response:
                if chunk.text:
                    full_answer.append(chunk.text)
                    yield chunk.text
            # Store the full answer for chat history
            stream_with_sources.full_answer = "".join(full_answer)

        gen = stream_with_sources()
        return {"stream": gen, "sources": sources, "get_answer": lambda: gen.full_answer}

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def total_chunks(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list:
        return list(self.doc_registry.values())
