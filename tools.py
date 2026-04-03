# tools.py  –  5 LangChain tools for the Schema Therapy Agent
#
# Tool 1: retrieve_from_books  — Agentic RAG (agent decides when to search books)
# Tool 2: find_technique        — Focused exercise generation from book excerpts
# Tool 3: search_memory         — Cross-session memory search
# Tool 4: save_session          — Session persistence with AI summary
# Tool 5: get_affirmation       — External API call (zenquotes.io)

import json
import os
from datetime import datetime
from pathlib import Path

import backoff
import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import RateLimitError, APIConnectionError
from pydantic import BaseModel, Field

SESSIONS_DIR     = Path("data/sessions")
MEMORY_DB_PATH   = "data/memory_vectorstore"
VECTORSTORE_PATH = "data/vectorstore"

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ── Shared helpers ────────────────────────────────────────────────────────────

@backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError), max_tries=4)
def _call_llm_with_backoff(llm_instance, messages):
    """Invoke LLM with exponential backoff on transient errors."""
    return llm_instance.invoke(messages)


def _get_memory_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="session_memory",
        embedding_function=embeddings,
        persist_directory=MEMORY_DB_PATH,
    )


# ── Pydantic input schemas ───────────────────────────────────────────────────

class RetrieveFromBooksInput(BaseModel):
    query: str = Field(
        description="Search query describing what to look up in the schema therapy books.",
        max_length=500,
    )

class FindTechniqueInput(BaseModel):
    situation: str = Field(
        description="Brief description of what the client is experiencing.",
        max_length=500,
    )

class SearchMemoryInput(BaseModel):
    query: str = Field(
        description="Short description of what you are looking for in past sessions.",
        max_length=500,
    )

class SaveSessionInput(BaseModel):
    reason: str = Field(
        description="Brief reason for saving (e.g. 'user said goodbye', 'session complete').",
        max_length=200,
    )

class GetAffirmationInput(BaseModel):
    theme: str = Field(
        description="Emotional theme to address (e.g. 'self-worth', 'resilience', 'letting go').",
        max_length=200,
    )


# ── Tool 1: retrieve_from_books (Agentic RAG) ────────────────────────────────

@tool(args_schema=RetrieveFromBooksInput)
def retrieve_from_books(query: str) -> str:
    """Search the schema therapy knowledge base for relevant passages from indexed books.
    Call this whenever you need to look up schema therapy concepts, modes, schemas,
    techniques, coping styles, or any clinical information.
    ALWAYS call this for clinical questions. Skip only for simple greetings or acknowledgements."""
    from rag import advanced_retrieve, book_is_loaded

    if not book_is_loaded():
        return "No books are indexed yet. Add PDFs to data/books/ and restart."

    docs, query_variants = advanced_retrieve(query, k=6)

    if not docs:
        return f"No relevant passages found for: {query}"

    passages = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        book = doc.metadata.get("source_book", doc.metadata.get("source", "book"))
        passages.append(f"[{book} p.{page}] {doc.page_content}")

    header = f"Found {len(passages)} relevant passages (search variants: {query_variants}):\n\n"
    return header + "\n\n---\n\n".join(passages)


# ── Tool 2: find_technique ────────────────────────────────────────────────────

@tool(args_schema=FindTechniqueInput)
def find_technique(situation: str) -> str:
    """Search the schema therapy books for a practical exercise or technique.
    Call when the user asks for exercises, techniques, or step-by-step practices.
    Input: brief description of what the client is experiencing."""
    if not os.path.exists(VECTORSTORE_PATH) or not os.listdir(VECTORSTORE_PATH):
        return "No books are indexed yet. Add PDFs to data/books/ and restart."

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    book_store = Chroma(
        collection_name="book_knowledge",
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_PATH,
    )

    results = book_store.similarity_search(
        f"exercise technique coping strategy for: {situation}", k=4
    )

    if not results:
        return "No specific technique found in the books for this situation."

    from prompts import load_prompts
    prompts = load_prompts()

    context = "\n\n---\n\n".join(doc.page_content for doc in results)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, request_timeout=30)
    prompt = prompts["find_technique_prompt"].format(situation=situation, context=context)
    response = _call_llm_with_backoff(llm, prompt)

    # Include source info
    sources = []
    for doc in results:
        book = doc.metadata.get("source_book", "book")
        page = doc.metadata.get("page", "?")
        sources.append(f"{book} p.{page}")

    return f"{response.content}\n\n(Sources: {', '.join(sources)})"


# ── Tool 3: search_memory ────────────────────────────────────────────────────

@tool(args_schema=SearchMemoryInput)
def search_memory(query: str) -> str:
    """Search past therapy session notes to find relevant history.
    Call when the user references the past ('remember', 'last time', 'before',
    'previous session') or when connecting current themes to earlier patterns."""
    try:
        memory_store = _get_memory_store()
        results = memory_store.similarity_search(query, k=3)
    except Exception:
        return "No past sessions found yet."

    if not results:
        return "No relevant past session notes found on this topic."

    parts = ["**Relevant memories from past sessions:**\n"]
    for i, doc in enumerate(results, 1):
        parts.append(f"**Memory {i}:**\n{doc.page_content}\n")
    return "\n".join(parts)


# ── Tool 4: save_session ─────────────────────────────────────────────────────

@tool(args_schema=SaveSessionInput)
def save_session(reason: str) -> str:
    """Save the current therapy session to disk and embed it into memory.
    Call when the user says goodbye, wants to save, or asks to end the session.
    Input: brief reason for saving."""
    import streamlit as st

    messages = st.session_state.get("messages", [])
    if not messages:
        return "Nothing to save — the conversation is empty."

    # Build conversation text from session state
    real_msgs = [
        m for m in messages
        if m.get("content") and not (
            m["role"] == "assistant"
            and "I'm glad you're here" in m.get("content", "")
        )
    ]
    if not real_msgs:
        return "Nothing meaningful to save yet."

    conversation = "\n".join(
        f"{'Client' if m['role'] == 'user' else 'Therapist'}: {m['content']}"
        for m in real_msgs
    )

    from prompts import load_prompts
    prompts = load_prompts()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, request_timeout=30)
    prompt = prompts["session_summary_prompt"].format(conversation=conversation[:6000])
    response = _call_llm_with_backoff(llm, prompt)
    raw = response.content.strip()

    # Strip markdown fences if model wraps JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        notes = json.loads(raw)
    except json.JSONDecodeError:
        notes = {
            "title": f"Session {datetime.now().strftime('%Y-%m-%d')}",
            "summary": raw[:500],
            "key_themes": [],
        }

    # Save to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in " _-" else "" for c in notes["title"])[:50]
    filepath = SESSIONS_DIR / f"{timestamp}_{safe_title}.json"
    filepath.write_text(json.dumps({
        "timestamp":    timestamp,
        "title":        notes["title"],
        "summary":      notes["summary"],
        "key_themes":   notes["key_themes"],
        "conversation": conversation,
    }, ensure_ascii=False, indent=2))

    # Embed into memory vectorstore for cross-session retrieval
    memory_store = _get_memory_store()
    summary_text = (
        f"Session: {notes['title']}\n"
        f"Date: {datetime.now().strftime('%B %d, %Y')}\n"
        f"Summary: {notes['summary']}\n"
        f"Themes: {', '.join(notes['key_themes'])}"
    )
    chunk_size = 1500
    conv_chunks = [conversation[i:i + chunk_size] for i in range(0, len(conversation), chunk_size)]
    all_texts = [summary_text] + conv_chunks
    all_meta = [{"timestamp": timestamp, "title": notes["title"], "type": "summary"}] + [
        {"timestamp": timestamp, "title": notes["title"], "type": "full_text", "chunk": i}
        for i in range(len(conv_chunks))
    ]
    memory_store.add_texts(texts=all_texts, metadatas=all_meta)

    st.session_state.session_saved = True

    return (
        f"Session saved as \"{notes['title']}\"\n\n"
        f"Summary: {notes['summary']}\n\n"
        f"Themes: {', '.join(notes['key_themes'])}"
    )


# ── Tool 5: get_affirmation (External API) ───────────────────────────────────

@tool(args_schema=GetAffirmationInput)
def get_affirmation(theme: str) -> str:
    """Fetch an inspirational quote for therapeutic encouragement.
    Call when the client seems discouraged, overwhelmed, or could benefit from
    an external source of motivation or a positive perspective shift.
    Use sparingly — at most once per conversation.
    Input: the emotional theme to address."""
    try:
        resp = requests.get("https://zenquotes.io/api/random", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            quote = data[0]
            return (
                f"\"{quote['q']}\" — {quote['a']}\n\n"
                f"(Source: zenquotes.io — a curated collection of inspirational quotes)\n\n"
                f"Reflect on how this connects to your experience with {theme}."
            )
    except Exception:
        pass

    # Fallback if API is unreachable
    return (
        f"Take a moment to breathe. Remember: your schemas don't define you. "
        f"You are learning new patterns around {theme}, and that takes courage. "
        f"Every time you notice a schema at work, you are already making progress."
    )


# ── All tools list (for dynamic toggling in the UI) ──────────────────────────

ALL_TOOLS = {
    "retrieve_from_books": retrieve_from_books,
    "find_technique":      find_technique,
    "search_memory":       search_memory,
    "save_session":        save_session,
    "get_affirmation":     get_affirmation,
}

TOOL_DESCRIPTIONS = {
    "retrieve_from_books": "Search therapy books",
    "find_technique":      "Find exercises",
    "search_memory":       "Search past sessions",
    "save_session":        "Save session",
    "get_affirmation":     "Inspirational quotes",
}
