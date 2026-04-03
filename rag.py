# rag.py  –  PDF ingestion + Advanced RAG with query translation, MMR, per-source cap

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import backoff
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from openai import RateLimitError, APIConnectionError

logger = logging.getLogger("schema_therapy_bot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ── Constants (no external config file needed) ────────────────────────────────
VECTORSTORE_PATH = Path("data/vectorstore")
BOOKS_DIR        = Path("data/books")
FINGERPRINT_FILE = VECTORSTORE_PATH / "fingerprint.json"
STATS_FILE       = VECTORSTORE_PATH / "index_stats.json"
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
EMBEDDING_MODEL  = "text-embedding-3-small"
CHAT_MODEL       = "gpt-4o-mini"

BOOKS_DIR.mkdir(parents=True, exist_ok=True)


@backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError), max_tries=4)
def _call_llm_with_backoff(llm_instance, messages):
    """Invoke LLM with exponential backoff on transient errors."""
    return llm_instance.invoke(messages)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def _get_store() -> Chroma:
    return Chroma(
        collection_name="book_knowledge",
        embedding_function=_get_embeddings(),
        persist_directory=str(VECTORSTORE_PATH),
    )

def _file_fingerprint(paths: list[Path]) -> str:
    """Hash filenames + sizes + mtimes to detect any book change."""
    h = hashlib.sha256()
    for p in sorted(paths):
        stat = p.stat()
        h.update(p.name.encode())
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()


# ── Rebuild detection ──────────────────────────────────────────────────────────

def needs_rebuild() -> bool:
    """True if books changed or vectorstore is missing."""
    pdfs = list(BOOKS_DIR.glob("*.pdf"))
    if not pdfs:
        return False
    new_fp = _file_fingerprint(pdfs)
    if not VECTORSTORE_PATH.exists() or not FINGERPRINT_FILE.exists():
        return True
    try:
        old = json.loads(FINGERPRINT_FILE.read_text())
        return old.get("fingerprint") != new_fp
    except Exception:
        return True


# ── Ingestion ──────────────────────────────────────────────────────────────────

def ingest_books_folder(force: bool = False) -> dict:
    """
    Ingest all PDFs from data/books/.
    Fingerprint-based: skips rebuild if nothing changed.
    Pass force=True to rebuild even if fingerprint matches (e.g. from UI button).
    Uses Chroma.from_documents which overwrites the collection in-place — no folder deletion needed.
    """
    pdf_files = list(BOOKS_DIR.glob("*.pdf"))
    if not pdf_files:
        # Clear stale stats so sidebar doesn't show old book info
        if STATS_FILE.exists():
            STATS_FILE.unlink()
        if FINGERPRINT_FILE.exists():
            FINGERPRINT_FILE.unlink()
        return {"books": 0, "pages": 0, "chunks": 0}

    if not force and not needs_rebuild():
        logger.info("Vectorstore up to date, skipping rebuild.")
        return {"books": len(pdf_files), "pages": 0, "chunks": 0, "cached": True}

    logger.info("Rebuilding vectorstore for %d books...", len(pdf_files))
    # Delete the existing collection before rebuild so removed books don't persist.
    # Deleting the collection (not the folder) avoids ChromaDB lock/readonly errors.
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(VECTORSTORE_PATH))
        existing = [c.name for c in client.list_collections()]
        if "book_knowledge" in existing:
            client.delete_collection("book_knowledge")
            logger.info("Deleted old book_knowledge collection before rebuild.")
    except Exception as e:
        logger.warning("Could not delete old collection (will overwrite): %s", e)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    all_chunks  = []
    total_pages = 0
    per_book    = {}

    failed_books = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages  = loader.load()
            for page in pages:
                page.metadata["source_book"] = pdf_path.stem
                page.metadata["source_file"] = pdf_path.name
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            total_pages += len(pages)
            per_book[pdf_path.name] = len(chunks)
            logger.info("  %s: %d pages, %d chunks", pdf_path.name, len(pages), len(chunks))
        except Exception as e:
            logger.error("  SKIPPED %s: %s", pdf_path.name, e)
            failed_books.append(pdf_path.name)
    if failed_books:
        logger.warning("Failed to load %d book(s): %s", len(failed_books), failed_books)
    if not all_chunks:
        raise RuntimeError(f"No chunks produced. All PDFs failed to load: {failed_books}")

    # Embed in batches to avoid OpenAI 400 errors from oversized payloads
    BATCH_SIZE = 100
    embeddings = _get_embeddings()

    # Create collection with first batch, then add the rest
    first_batch = all_chunks[:BATCH_SIZE]
    db = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        collection_name="book_knowledge",
        persist_directory=str(VECTORSTORE_PATH),
    )
    for i in range(BATCH_SIZE, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        db.add_documents(batch)
        logger.info("  Embedded chunks %d-%d / %d", i, min(i + BATCH_SIZE, len(all_chunks)), len(all_chunks))

    FINGERPRINT_FILE.write_text(
        json.dumps({"fingerprint": _file_fingerprint(pdf_files)}, indent=2)
    )
    STATS_FILE.write_text(json.dumps(per_book, indent=2))

    logger.info("Done: %d total chunks", len(all_chunks))
    return {"books": len(pdf_files), "pages": total_pages, "chunks": len(all_chunks)}


def read_index_stats() -> dict:
    if not STATS_FILE.exists():
        return {}
    try:
        return json.loads(STATS_FILE.read_text())
    except Exception:
        return {}


# ── Query translation (Advanced RAG) ──────────────────────────────────────────

def translate_query(user_query: str) -> list[str]:
    """
    Rewrite the user query into 3 search variants using schema therapy vocabulary.
    Query translation improves retrieval recall significantly.
    """
    from prompts import load_prompts
    prompts = load_prompts()

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.0, request_timeout=30)
    prompt = prompts["query_translation_prompt"].format(user_query=user_query)
    try:
        resp = _call_llm_with_backoff(llm, prompt)
        arr  = json.loads(resp.content)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr[:3]
    except Exception:
        pass
    return [user_query]


# ── Advanced retrieval ─────────────────────────────────────────────────────────

def advanced_retrieve(user_query: str, k: int = 5) -> tuple[list[Any], list[str]]:
    """
    Advanced retrieval combining:
    - Query translation: multiple reformulations for better recall
    - MMR (Maximal Marginal Relevance): diverse, non-redundant results
    - Per-source cap: no single book dominates the context
    """
    if not book_is_loaded():
        return [], [user_query]

    store   = _get_store()
    queries = translate_query(user_query)

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           k,
            "fetch_k":     max(30, k * 5),
            "lambda_mult": 0.5,
        },
    )

    seen         = set()
    all_docs     = []
    source_count = {}

    for query in queries:
        try:
            docs = retriever.invoke(query)
            for doc in docs:
                key = doc.page_content[:120]
                if key in seen:
                    continue
                src = doc.metadata.get("source_book", "unknown")
                if source_count.get(src, 0) >= 4:
                    continue
                seen.add(key)
                source_count[src] = source_count.get(src, 0) + 1
                all_docs.append(doc)
        except Exception as e:
            logger.warning("Retrieval error for query '%s': %s", query, e)

    logger.info("Retrieved %d chunks from: %s", len(all_docs[:k]), list(source_count.keys()))
    return all_docs[:k], queries


# ── Status helpers ─────────────────────────────────────────────────────────────

def book_is_loaded() -> bool:
    return VECTORSTORE_PATH.exists() and any(VECTORSTORE_PATH.iterdir())

def get_loaded_books() -> list[str]:
    return [f.stem for f in BOOKS_DIR.glob("*.pdf")]