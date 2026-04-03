# agent.py  –  LangGraph ReAct Agent for Schema Therapy
#
# This module creates a proper ReAct (Reason + Act) agent using LangGraph.
# The agent decides autonomously which tools to call and when to stop —
# this is the key difference from a fixed pipeline.
#
# Architecture (ReAct loop):
#   User message → Agent reasons → Tool call? → Observe result → Reason again → ... → Final response
#
# The agent is NOT a pipeline. It dynamically decides:
#   - Whether to search books (agentic RAG)
#   - Which tools to call and in what order
#   - When it has enough information to respond
#   - When to stop the loop

from datetime import datetime

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from feedback import get_feedback_insights
from prompts import load_prompts

# Try to import Anthropic (optional dependency)
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(provider: str, model: str, temperature: float = 0.6):
    """
    Create an LLM instance for the given provider and model.
    Supports OpenAI and Anthropic with retry logic built in.
    """
    if provider == "OpenAI":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            request_timeout=30,
            max_retries=3,
        )
    elif provider == "Anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "langchain-anthropic is not installed. "
                "Run: pip install langchain-anthropic"
            )
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            timeout=30,
            max_retries=3,
        )
    raise ValueError(f"Unknown provider: {provider}")


# ── Singleton checkpointer ───────────────────────────────────────────────────
# MemorySaver persists conversation state per thread_id.
# Cached as a singleton so it survives Streamlit reruns.

_checkpointer = None

def get_checkpointer() -> MemorySaver:
    """Return the singleton MemorySaver instance."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_system_prompt(personality: str = "warm") -> str:
    """
    Build the full system prompt from YAML templates + personality + feedback insights.
    This is called before each agent invocation to ensure the prompt is up to date.
    """
    prompts = load_prompts()

    # Base system prompt with today's date
    base = prompts["system_prompt"].format(date=datetime.now().strftime("%B %d, %Y"))

    # Personality modifier
    personality_key = f"personality_{personality}"
    personality_text = prompts.get(personality_key, "")

    # Learning agent: feedback insights shape future responses
    feedback_text = get_feedback_insights()

    parts = [base]
    if personality_text:
        parts.append(personality_text)
    if feedback_text:
        parts.append(feedback_text)

    return "\n\n".join(parts)


# ── Language detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect language of user message with Lithuanian-first heuristics."""
    # Lithuanian special characters are reliable
    if any(c in "ąčęėįšųūžĄČĘĖĮŠŲŪŽ" for c in text):
        return "Lithuanian"

    lt_words = {
        "aš", "jis", "ji", "mes", "jūs", "yra", "buvo", "kaip", "ką",
        "labai", "taip", "dėl", "noriu", "galiu", "žinau", "kodėl",
        "ar", "prisimeni", "kokia", "mano", "spalva", "kas", "apie",
        "kada", "kuris", "kuri", "jeigu", "nes", "bet", "arba",
        "norėčiau", "gali", "negali", "nežinau", "suprantu", "jaučiu",
        "jaučiuosi", "manau", "galvoju", "sakau", "klausiu",
    }
    words = set(text.lower().replace("?", "").replace(".", "").replace(",", "").split())
    if len(words & lt_words) >= 1:
        return "Lithuanian"

    # Short messages (< 4 words) are unreliable for langdetect — default to English
    # "Hi", "Hello", "Thanks", "Ok" etc. were being misdetected as Dutch, Malay, etc.
    if len(words) < 4:
        return "English"

    try:
        from langdetect import detect
        code = detect(text)
        lang_names = {
            "lt": "Lithuanian", "en": "English", "de": "German",
            "fr": "French", "es": "Spanish", "it": "Italian",
            "pl": "Polish", "ru": "Russian", "pt": "Portuguese",
            "nl": "Dutch", "sv": "Swedish", "no": "Norwegian",
            "da": "Danish", "fi": "Finnish", "lv": "Latvian",
            "et": "Estonian", "uk": "Ukrainian", "cs": "Czech",
        }
        return lang_names.get(code, "English")
    except Exception:
        return "English"


# ── Agent factory ─────────────────────────────────────────────────────────────

def create_therapy_agent(
    provider: str,
    model: str,
    temperature: float,
    tools: list,
    personality: str = "warm",
):
    """
    Create a LangGraph ReAct agent with the given configuration.

    This is the core of the Sprint 3 upgrade:
    - The agent uses the ReAct pattern (Reason + Act)
    - It decides which tools to call based on the conversation
    - Retrieval is a TOOL the agent chooses to use (agentic RAG)
    - The checkpointer gives the agent memory within a session
    """
    llm = get_llm(provider, model, temperature)
    checkpointer = get_checkpointer()
    system_prompt = build_system_prompt(personality)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        prompt=system_prompt,
    )

    return agent
