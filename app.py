# app.py  –  Schema Therapy Agent  (Streamlit + LangGraph ReAct)
#
# Sprint 3 upgrade: proper ReAct agent with agentic RAG, 5 tools,
# multi-model support, personality selector, feedback loop, tool toggles.

import json
import os
import re
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from agent import (
    ANTHROPIC_AVAILABLE,
    create_therapy_agent,
    detect_language,
    get_checkpointer,
)
from costs import (
    MODEL_DISPLAY_NAMES,
    PROVIDER_MODELS,
    calculate_cost,
    format_cost,
)
from feedback import get_feedback_insights, load_feedback, save_feedback_entry
from rag import (
    book_is_loaded,
    get_loaded_books,
    ingest_books_folder,
    read_index_stats,
)
from tools import ALL_TOOLS, TOOL_DESCRIPTIONS

load_dotenv()


# ── Security & validation ────────────────────────────────────────────────────
MAX_MESSAGE_LENGTH   = 2000
MAX_MESSAGES_SESSION = 50
RATE_LIMIT_SECONDS   = 5


def _normalise(text: str) -> str:
    """Normalise text for prompt-injection detection."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[\s\W]+", " ", text.lower())
    return text


def validate_input(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "Please enter a message."
    if len(text) > MAX_MESSAGE_LENGTH:
        return False, f"Message too long ({len(text)} chars). Keep it under {MAX_MESSAGE_LENGTH}."
    normalised = _normalise(text)
    suspicious = [
        "ignore previous instructions", "ignore all instructions",
        "you are now", "disregard your instructions",
        "forget your system prompt", "override your constraints",
        "new persona", "act as", "jailbreak",
        "discard your rules",
    ]
    if any(p in normalised for p in suspicious):
        return False, "I noticed something unusual in your message. Please rephrase."
    return True, ""


def check_rate_limit() -> bool:
    last = st.session_state.get("last_message_time")
    if last is None:
        return True
    return (datetime.now() - last).total_seconds() >= RATE_LIMIT_SECONDS


def repair_chat_history(thread_id: str) -> None:
    """Fix corrupted chat history where AIMessages have tool_calls without
    corresponding ToolMessages. This can happen when rapid messages interrupt
    a tool call mid-stream. Adds placeholder ToolMessages so the session recovers."""
    checkpointer = get_checkpointer()
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_tuple = checkpointer.get_tuple(config)
    if checkpoint_tuple is None:
        return

    checkpoint = checkpoint_tuple.checkpoint
    messages = checkpoint.get("channel_values", {}).get("messages", [])
    if not messages:
        return

    repaired = []
    needs_repair = False
    for msg in messages:
        repaired.append(msg)
        # If this is an AI message with tool calls, check if next message is a ToolMessage
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tc_id = tc.get("id", "unknown")
                # Check if a matching ToolMessage already follows
                has_result = any(
                    isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) == tc_id
                    for m in messages[messages.index(msg) + 1:]
                )
                if not has_result:
                    needs_repair = True
                    repaired.append(ToolMessage(
                        content="[Tool call interrupted — no result available]",
                        tool_call_id=tc_id,
                        name=tc.get("name", "unknown"),
                    ))

    if needs_repair:
        checkpoint["channel_values"]["messages"] = repaired
        checkpointer.put(config, checkpoint, checkpoint_tuple.metadata, checkpoint_tuple.parent_config)


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Schema Therapy Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] {
    background-color: #f9f9f9 !important;
    border-right: 1px solid #e5e5e5 !important;
}
[data-testid="stSidebar"] * { color: #1a1a1a !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    color: #bbb !important;
    margin-bottom: 0.5rem !important;
}
.main .block-container {
    max-width: 740px !important;
    padding: 0 2rem 8rem !important;
    margin: 0 auto !important;
}

/* Title */
.app-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1a1a1a;
    padding: 1.6rem 0 0.3rem;
    letter-spacing: -0.02em;
}
.app-subtitle {
    font-size: 0.82rem;
    color: #999;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #efefef;
    margin-bottom: 1.6rem;
}

/* Buttons */
.stButton > button {
    background: #fff !important;
    color: #1a1a1a !important;
    border: 1px solid #ddd !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #f5f5f5 !important;
    border-color: #bbb !important;
}

/* Cost rows */
.cost-row {
    display: flex;
    justify-content: space-between;
    padding: 0.38rem 0;
    border-bottom: 1px solid #f2f2f2;
    font-size: 0.79rem;
}
.cost-label { color: #999; }
.cost-value { font-weight: 500; color: #1a1a1a; }

/* Pills */
.pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
}
.pill-green { background: #f0faf0; color: #2d8a2d; border: 1px solid #c3e6c3; }
.pill-gray  { background: #f5f5f5; color: #999;    border: 1px solid #e5e5e5; }

hr { border-color: #f0f0f0 !important; margin: 0.85rem 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #e0e0e0; border-radius: 4px; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Auto-load books ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def auto_load_books():
    return ingest_books_folder()

with st.spinner("Loading knowledge base..."):
    auto_load_books()


# ── Session state ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "session_cost": 0.0,
        "session_tokens": 0,
        "last_prompt_cost": 0.0,
        "last_prompt_tokens": 0,
        "session_saved": False,
        "last_message_time": None,
        "thread_id": str(uuid.uuid4()),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Session controls (top — most important) ───────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save & End", use_container_width=True):
            if st.session_state.messages:
                with st.spinner("Saving..."):
                    from tools import save_session
                    result = save_session.invoke({"reason": "User clicked Save & End"})
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": result + "\n\n---\n*Session ended. Click **New Session** to start a new conversation.*",
                    }]
                st.rerun()
            else:
                st.warning("Nothing to save.")
    with col2:
        if st.button("New Session", use_container_width=True):
            for key in ["messages", "session_cost", "session_tokens",
                        "last_prompt_cost", "last_prompt_tokens",
                        "session_saved", "last_message_time"]:
                if key == "messages":
                    st.session_state[key] = []
                elif "cost" in key or "tokens" in key:
                    st.session_state[key] = 0.0 if "cost" in key else 0
                elif key == "last_message_time":
                    st.session_state[key] = None
                else:
                    st.session_state[key] = False
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    st.markdown("---")

    # ── Settings (collapsed by default — keeps sidebar clean) ─────────────
    with st.expander("Settings", expanded=False):
        # Model
        available_providers = ["OpenAI"]
        if os.environ.get("ANTHROPIC_API_KEY") and ANTHROPIC_AVAILABLE:
            available_providers.append("Anthropic")

        provider = st.selectbox(
            "Provider",
            available_providers,
            key="provider",
        )

        model_options = PROVIDER_MODELS.get(provider, ["gpt-4o-mini"])
        model_display = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_options]
        model_idx = st.selectbox(
            "Model",
            range(len(model_options)),
            format_func=lambda i: model_display[i],
            key="model_idx",
        )
        selected_model = model_options[model_idx]

        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.caption("Add `ANTHROPIC_API_KEY` to .env for Anthropic.")

        # Personality
        personality_options = {
            "Warm & Supportive": "warm",
            "Professional & Clinical": "professional",
            "Concise & Direct": "concise",
        }
        personality_label = st.selectbox(
            "Response style",
            list(personality_options.keys()),
            key="personality_label",
        )
        selected_personality = personality_options[personality_label]

    # Read settings from expander (defaults if never opened)
    if "provider" not in st.session_state:
        provider = "OpenAI"
        selected_model = "gpt-4o-mini"
        selected_personality = "warm"
    else:
        provider = st.session_state.provider
        model_options = PROVIDER_MODELS.get(provider, ["gpt-4o-mini"])
        model_idx_val = st.session_state.get("model_idx", 0)
        selected_model = model_options[model_idx_val] if model_idx_val < len(model_options) else model_options[0]
        personality_label = st.session_state.get("personality_label", "Warm & Supportive")
        personality_options = {
            "Warm & Supportive": "warm",
            "Professional & Clinical": "professional",
            "Concise & Direct": "concise",
        }
        selected_personality = personality_options.get(personality_label, "warm")

    # ── Tools (collapsed by default) ──────────────────────────────────────
    with st.expander("Tools", expanded=False):
        tool_states = {}
        for tool_name, description in TOOL_DESCRIPTIONS.items():
            if tool_name == "save_session":
                tool_states[tool_name] = True
                st.checkbox(description, value=True, disabled=True, key=f"tool_{tool_name}")
            else:
                tool_states[tool_name] = st.checkbox(
                    description, value=True, key=f"tool_{tool_name}",
                )

    # Read tool states (defaults if never opened)
    if "tool_retrieve_from_books" not in st.session_state:
        tool_states = {name: True for name in TOOL_DESCRIPTIONS}
    else:
        tool_states = {}
        for tool_name in TOOL_DESCRIPTIONS:
            if tool_name == "save_session":
                tool_states[tool_name] = True
            else:
                tool_states[tool_name] = st.session_state.get(f"tool_{tool_name}", True)

    st.markdown("---")

    # ── Usage & Cost (always visible — compact) ──────────────────────────
    st.markdown("### Usage")
    st.markdown(f"""
    <div class="cost-row">
        <span class="cost-label">Model</span>
        <span class="cost-value">{MODEL_DISPLAY_NAMES.get(selected_model, selected_model)}</span>
    </div>
    <div class="cost-row">
        <span class="cost-label">Last message</span>
        <span class="cost-value">{st.session_state.last_prompt_tokens:,} tok · {format_cost(st.session_state.last_prompt_cost)}</span>
    </div>
    <div class="cost-row" style="border:none">
        <span class="cost-label">Session total</span>
        <span class="cost-value">{st.session_state.session_tokens:,} tok · {format_cost(st.session_state.session_cost)}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Knowledge base (compact) ──────────────────────────────────────────
    st.markdown("### Books")
    loaded_books = get_loaded_books()
    if loaded_books:
        for b in loaded_books:
            st.markdown(
                f'<span class="pill pill-green">✓</span> '
                f'<span style="font-size:0.75rem;color:#555">{b[:32]}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Add PDFs to data/books/")

    if st.button("Update Index", use_container_width=True):
        if loaded_books:
            with st.spinner("Indexing..."):
                try:
                    result = ingest_books_folder(force=True)
                    st.cache_resource.clear()
                    st.success(f"✓ {result.get('books', 0)} book(s) indexed.")
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.warning("No PDFs found.")

    st.markdown("---")

    # ── Past sessions (compact — just titles and dates) ───────────────────
    st.markdown("### Past Sessions")
    sessions_dir = Path("data/sessions")
    if sessions_dir.exists():
        session_files = sorted(sessions_dir.glob("*.json"), reverse=True)[:6]
        if session_files:
            for sf in session_files:
                try:
                    data = json.loads(sf.read_text())
                    ts = data.get("timestamp", "")[:8]
                    if len(ts) == 8:
                        ts = f"{ts[:4]}-{ts[4:6]}-{ts[6:]}"
                    title = data.get("title", sf.stem)[:45]
                    themes = data.get("key_themes", [])
                    theme_str = f" · {', '.join(themes[:2])}" if themes else ""
                    st.caption(f"**{title}** — {ts}{theme_str}")
                except Exception:
                    pass
        else:
            st.caption("No saved sessions yet.")
    else:
        st.caption("No saved sessions yet.")

    # ── Feedback stats (only if exists) ───────────────────────────────────
    feedback_entries = load_feedback()
    if feedback_entries:
        st.markdown("---")
        recent = feedback_entries[-20:]
        pos = sum(1 for e in recent if e["rating"] == "positive")
        neg = len(recent) - pos
        st.caption(f"Feedback: {pos} 👍 / {neg} 👎")

    # ── Help (at the very bottom) ─────────────────────────────────────────
    st.markdown("---")
    with st.expander("Help", expanded=False):
        st.markdown("""
**Example messages:**
- *"What is the abandonment schema?"*
- *"I feel like nobody really cares about me"*
- *"Give me an exercise for emotional deprivation"*
- *"Do you remember what we talked about last time?"*

**Tips:**
- Write in any language — I'll respond in yours
- Use 👍/👎 to rate responses — I learn from feedback
- Open **Settings** to switch models or personality
- Open **Tools** to enable/disable agent capabilities
        """)


# ── MAIN CHAT ────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🧠 Schema Therapy Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Evidence-based schema therapy · ReAct agent · Agentic RAG · Multilingual'
    '</div>',
    unsafe_allow_html=True,
)

# Session length warning
msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
if msg_count >= MAX_MESSAGES_SESSION:
    st.warning(f"You've sent {msg_count} messages. Consider saving and starting a new session.")

# Welcome message
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello. I'm glad you're here.\n\n"
            "This is a space to explore what's on your mind, at whatever pace feels right. "
            "I work within schema therapy — interested in the patterns and experiences that shape "
            "how you feel and relate to others.\n\n"
            "You can write in any language and I'll respond in yours. "
            "What would you like to talk about today?"
        ),
    })

# ── Render messages with feedback buttons ─────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    with st.chat_message(role, avatar="🧠" if role == "assistant" else "👤"):
        st.markdown(msg["content"])

        if role == "assistant":
            # Show metadata
            tokens_info = msg.get("tokens_info", "")
            sources = msg.get("sources", [])
            tools_used = msg.get("tools_used", [])

            if tokens_info:
                st.caption(tokens_info)
            if sources:
                source_str = " · ".join(f"*{s['book']}* p.{s['page']}" for s in sources)
                st.caption(f"📖 Sources: {source_str}")
            if tools_used:
                st.caption(f"🔧 Tools: {', '.join(tools_used)}")

            # Feedback buttons (skip welcome message and session-end messages)
            if i > 0 and "Session ended" not in msg.get("content", ""):
                fb_key = f"feedback_{i}_{st.session_state.thread_id}"
                feedback = st.feedback("thumbs", key=fb_key)
                if feedback is not None:
                    # Only save if we haven't already recorded feedback for this message
                    saved_key = f"feedback_saved_{i}_{st.session_state.thread_id}"
                    if saved_key not in st.session_state:
                        rating = "positive" if feedback == 1 else "negative"
                        save_feedback_entry(i, rating, msg.get("content", "")[:300])
                        st.session_state[saved_key] = True


# ── Input handling ────────────────────────────────────────────────────────────
if not book_is_loaded():
    st.caption("⚠️ No books found. Add PDFs to data/books/ and restart.")

if st.session_state.get("session_saved"):
    st.info("Session ended. Click **New Session** in the sidebar to start a new conversation.")
    user_input = None
else:
    user_input = st.chat_input("Write a message...")

if user_input and user_input.strip():
    # Rate limit
    if not check_rate_limit():
        remaining = RATE_LIMIT_SECONDS - (datetime.now() - st.session_state.last_message_time).total_seconds()
        st.warning(f"Please wait {int(remaining) + 1} seconds before sending another message.")
        st.stop()

    # Validate
    is_valid, error_msg = validate_input(user_input)
    if not is_valid:
        st.error(error_msg)
        st.stop()

    user_text = user_input.strip()

    # Update rate-limit timestamp immediately (before processing) to block rapid re-submissions
    st.session_state.last_message_time = datetime.now()

    # Add language instruction to the user message so the agent responds in the right language
    detected_lang = detect_language(user_text)
    lang_suffix = (
        f"\n\n[SYSTEM NOTE — not visible to the user: "
        f"The user's message is in {detected_lang}. "
        f"You MUST respond ENTIRELY in {detected_lang}. "
        f"Do NOT write even one sentence in any other language.]"
    )

    # Add to session state for UI
    st.session_state.messages.append({"role": "user", "content": user_text})

    # Render user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_text)

    # ── Build and invoke the ReAct agent ──────────────────────────────────
    # Collect active tools based on sidebar toggles
    active_tools = [
        ALL_TOOLS[name]
        for name, enabled in tool_states.items()
        if enabled and name in ALL_TOOLS
    ]

    # Create agent with current settings
    agent = create_therapy_agent(
        provider=provider,
        model=selected_model,
        temperature=0.6,
        tools=active_tools,
        personality=selected_personality,
    )

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Repair any corrupted chat history (e.g. from interrupted tool calls)
    repair_chat_history(st.session_state.thread_id)

    # Invoke the agent with streaming
    with st.chat_message("assistant", avatar="🧠"):
        full_response = ""
        tools_used = []
        sources = []
        total_tokens = 0
        total_cost = 0.0

        status_placeholder = st.empty()
        response_placeholder = st.empty()

        try:
            status_placeholder.caption("🔄 Thinking...")

            # Stream the agent's response
            for event in agent.stream(
                {"messages": [HumanMessage(content=user_text + lang_suffix)]},
                config,
                stream_mode="messages",
            ):
                msg_chunk, metadata = event
                node = metadata.get("langgraph_node", "")

                # Tool results — extract tool names and sources
                if isinstance(msg_chunk, ToolMessage):
                    tool_name = msg_chunk.name or "tool"
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    status_placeholder.caption(
                        f"🔧 Using: {', '.join(tools_used)}..."
                    )
                    # Parse sources from retrieve_from_books results
                    if tool_name == "retrieve_from_books" and msg_chunk.content:
                        import re as _re
                        for match in _re.finditer(r"\[(.+?)\s+p\.(\d+)\]", msg_chunk.content):
                            sources.append({"book": match.group(1), "page": match.group(2)})

                # AI message chunks — stream the final response
                elif isinstance(msg_chunk, AIMessageChunk) and node == "agent":
                    # Only stream text content (not tool-call tokens)
                    if msg_chunk.content and not (
                        hasattr(msg_chunk, "tool_call_chunks") and msg_chunk.tool_call_chunks
                    ):
                        full_response += msg_chunk.content
                        response_placeholder.markdown(full_response + "▌")

            # Finalise display
            status_placeholder.empty()
            if full_response:
                response_placeholder.markdown(full_response)
            else:
                response_placeholder.markdown(
                    "I'm sorry, something went wrong. Please try again."
                )

        except Exception as e:
            status_placeholder.empty()
            error_str = str(e).lower()
            if "credit balance" in error_str or "billing" in error_str or "insufficient" in error_str:
                full_response = "⚠️ The API credit balance is too low. Please switch to a different provider in Settings, or add credits to your account."
            elif "rate limit" in error_str or "too many requests" in error_str:
                full_response = "⚠️ The API is temporarily overloaded. Please wait a moment and try again."
            elif "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
                full_response = "⚠️ There's an issue with the API key. Please check your `.env` file and restart the app."
            elif "invalid_chat_history" in error_str or "toolmessage" in error_str:
                full_response = "⚠️ The conversation history got corrupted. Please click **New Session** in the sidebar to start fresh."
            else:
                full_response = f"Something went wrong. Please try again or start a new session."
            response_placeholder.markdown(full_response)

        # ── Cost tracking ─────────────────────────────────────────────────
        # Estimate tokens from response length (accurate tracking requires
        # usage_metadata which is available on invoke but not stream chunks)
        est_input_tokens = len(user_text.split()) * 2 + 500  # rough estimate
        est_output_tokens = len(full_response.split()) * 2
        total_tokens = est_input_tokens + est_output_tokens
        total_cost = calculate_cost(selected_model, est_input_tokens, est_output_tokens)

        tokens_info = f"~{total_tokens:,} tokens · {format_cost(total_cost)}"
        st.caption(tokens_info)
        if sources:
            # Deduplicate sources
            seen = set()
            unique_sources = []
            for s in sources:
                key = f"{s['book']}_p{s['page']}"
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(s)
            source_str = " · ".join(f"*{s['book']}* p.{s['page']}" for s in unique_sources[:8])
            st.caption(f"📖 Sources: {source_str}")
            sources = unique_sources[:8]
        if tools_used:
            st.caption(f"🔧 Tools: {', '.join(tools_used)}")

        # Feedback button for this new response
        fb_key = f"feedback_{len(st.session_state.messages)}_{st.session_state.thread_id}"
        feedback = st.feedback("thumbs", key=fb_key)
        if feedback is not None:
            saved_key = f"feedback_saved_{len(st.session_state.messages)}_{st.session_state.thread_id}"
            if saved_key not in st.session_state:
                rating = "positive" if feedback == 1 else "negative"
                save_feedback_entry(len(st.session_state.messages), rating, full_response[:300])
                st.session_state[saved_key] = True

    # ── Update session state ──────────────────────────────────────────────
    # Note: last_message_time is already set at the start (before processing)
    # to properly block rapid re-submissions via rate limiting.
    st.session_state.last_prompt_cost = total_cost
    st.session_state.last_prompt_tokens = total_tokens
    st.session_state.session_cost += total_cost
    st.session_state.session_tokens += total_tokens

    st.session_state.messages.append({
        "role":        "assistant",
        "content":     full_response,
        "tokens_info": tokens_info,
        "sources":     sources,
        "tools_used":  tools_used,
    })

    # Rerun so sidebar updates
    st.rerun()
