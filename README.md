# Schema Therapy Agent — Sprint 3

## Problem Definition

**What problem does this solve?**

Schema therapy is a powerful psychotherapy model, but accessing its concepts, identifying personal schemas, and finding practical exercises requires reading dense clinical textbooks. Most people who could benefit from schema therapy insights don't have the time or background to read these books.

**How does this app address it?**

This agent acts as a schema therapy coach — a conversational AI that has "studied" the books and can explain concepts, identify schemas and modes, suggest exercises, and remember past conversations. It makes evidence-based psychological knowledge accessible and personalised.

**Target users:**
- People exploring schema therapy for personal growth
- Therapy clients who want to understand their patterns between sessions
- Students learning schema therapy concepts
- Anyone curious about emotional patterns and coping styles

---

## What This Agent Does

- **Answers questions** about schema therapy, early maladaptive schemas, and modes using content retrieved directly from indexed books
- **Identifies schemas and modes** in what the user describes — naming the Vulnerable Child, Detached Protector, etc. and explaining what it means for their specific situation
- **Suggests practical exercises** grounded in the literature, with step-by-step instructions
- **Remembers past sessions** and connects themes across conversations
- **Responds in any language** the user writes in (55 languages via langdetect)
- **Learns from feedback** — adapts its style based on 👍/👎 ratings over time
- **Supports multiple LLMs** — OpenAI and Anthropic models, switchable in the UI

---

## Architecture: ReAct Agent (Sprint 3)

This is a **LangGraph ReAct agent**, not a fixed pipeline. The key difference:

| Aspect | Pipeline (Sprint 2) | Agent (Sprint 3) |
|---|---|---|
| Control flow | Developer decides each step | Agent decides dynamically |
| Retrieval | Always happens, every message | Agent decides when to search books |
| Tool usage | Fixed, one round | Agent can call multiple tools, multiple rounds |
| Stopping | After one response | Agent decides when it has enough info |

**The ReAct loop:**
```
User message → Agent reasons → Need book info? → retrieve_from_books
                             → Need an exercise? → find_technique
                             → Past session reference? → search_memory
                             → Client needs encouragement? → get_affirmation
                             → Done? → Final response
```

The agent is built with `create_react_agent` from LangGraph, which implements the Reason + Act pattern: the LLM alternates between reasoning about what to do next and acting via tool calls, observing results, and deciding whether to continue or respond.

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Agent framework | LangGraph (ReAct agent) |
| LLM (default) | OpenAI GPT-4o-mini |
| LLM (alternative) | Anthropic Claude Sonnet 4 / Haiku 4 |
| Embeddings | OpenAI text-embedding-3-small |
| Vector database | ChromaDB (local, persistent) |
| LLM framework | LangChain |
| Memory | LangGraph MemorySaver (within-session) + ChromaDB (cross-session) |
| Observability | LangSmith |
| PDF loading | LangChain PyPDFLoader |
| Language detection | langdetect |
| External API | zenquotes.io (affirmations) |

---

## Project Structure

```
app.py            — Streamlit UI, agent invocation, streaming, feedback
agent.py          — LangGraph ReAct agent factory, LLM factory, prompt builder
tools.py          — 5 LangChain tools with Pydantic validation
rag.py            — PDF ingestion, chunking, embedding, advanced retrieval
costs.py          — Multi-model token usage and cost calculation
feedback.py       — Feedback storage and analysis (learning agent)
prompts.py        — YAML prompt loader
prompts.yaml      — Externalised system/tool/personality prompts
pyproject.toml    — Project metadata and dependencies

data/
  books/              — Schema therapy PDF books (add your own)
  sessions/           — Saved session JSON files (auto-created)
  vectorstore/        — ChromaDB book embeddings (auto-created)
  memory_vectorstore/ — ChromaDB session memory (auto-created)
  feedback.json       — User feedback ratings (auto-created)

.env              — API keys (see Setup)
```

---

## Setup

**Requirements:** Python 3.10+

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install streamlit langchain langchain-openai langchain-anthropic \
    langchain-community langchain-chroma langgraph langgraph-prebuilt \
    langgraph-checkpoint chromadb pypdf python-dotenv langsmith \
    langchain-text-splitters langdetect backoff pyyaml requests pydantic
```

**Create a `.env` file** in the project root:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=schema-therapy-agent
```

> Note: `ANTHROPIC_API_KEY` is optional. Without it, only OpenAI models are available.

**Add your books:** drop schema therapy PDF files into `data/books/`.

**Run:**
```bash
streamlit run app.py
```

---

## The 5 Tools

| # | Tool | Type | What it does |
|---|---|---|---|
| 1 | `retrieve_from_books` | Agentic RAG | Searches indexed books for relevant passages. The agent decides when to call this. |
| 2 | `find_technique` | Domain tool | Finds practical exercises from the books with step-by-step instructions. |
| 3 | `search_memory` | Memory tool | Searches past session notes for cross-session context. |
| 4 | `save_session` | Persistence tool | Saves the session with AI-generated title, summary, and themes. |
| 5 | `get_affirmation` | External API | Fetches inspirational quotes from zenquotes.io for encouragement. |

All tools have **Pydantic input schemas** for field-level validation (max_length, descriptions). Tools are **dynamically toggleable** — users can enable/disable tools via sidebar checkboxes.

---

## How Agentic RAG Works (Hard #1)

In a normal RAG pipeline, retrieval happens on every message. In **agentic RAG**, the agent decides when to search:

1. User says "hello" → Agent responds directly (no retrieval needed)
2. User asks "what is the abandonment schema?" → Agent calls `retrieve_from_books` → reads passages → responds grounded in books
3. User asks "give me an exercise" → Agent calls `retrieve_from_books` AND `find_technique` → combines both results

The retrieval tool uses **advanced RAG** internally:
- **Query translation**: rewrites the query into 3 clinical variants via LLM
- **MMR retrieval**: Maximal Marginal Relevance for diverse, non-redundant results
- **Per-source cap**: max 4 chunks from any single book

---

## How the Learning Agent Works (Hard #4)

1. Every assistant response has 👍/👎 feedback buttons
2. Ratings are stored in `data/feedback.json` with response snippets
3. Before each agent call, `get_feedback_insights()` analyses recent feedback
4. Insights are injected into the system prompt: "User prefers concise responses", "User appreciates practical exercises"
5. The agent adapts its communication style over time

This implements the **learning agent** pattern from Sprint 3: performance element (agent), critic (user feedback), learning element (feedback analysis), and the loop between them.

---

## Optional Tasks Implemented

### Easy
- **#1** — Critique: Project 2 was reviewed and all 9 feedback areas were addressed
- **#2** — Personality: sidebar dropdown (Warm & Supportive / Professional & Clinical / Concise & Direct)
- **#3** — LLM choice: sidebar dropdown to choose between OpenAI and Anthropic models
- **#5** — Help guide: expandable "How to use this bot" section in sidebar

### Medium
- **#1** — Token usage and cost display: per-message and session totals, multi-model pricing
- **#2** — Retry logic: exponential backoff on all LLM/API calls, max_retries on model instances
- **#3** — Memory: LangGraph MemorySaver (within-session) + ChromaDB vectorstore (cross-session)
- **#4** — External API tool: `get_affirmation` calls zenquotes.io for motivational quotes
- **#7** — Feedback loop: 👍/👎 buttons, stored ratings, feedback insights in system prompt
- **#8** — 5 tools with enable/disable toggles in the sidebar
- **#9** — Multi-model: OpenAI (GPT-4o-mini, GPT-4o) + Anthropic (Claude Sonnet 4, Haiku 4)

### Hard
- **#1** — Agentic RAG: retrieval is a tool the agent decides to use, not a fixed pipeline step
- **#2** — LLM observability: LangSmith tracing for all agent interactions
- **#4** — Learning from feedback: agent adapts style based on accumulated user ratings

**Total: 4 easy + 7 medium + 3 hard**

---

## Security

- **Input length limit** — messages over 2000 characters are rejected
- **Prompt injection detection** — NFKD normalisation + suspicious pattern matching
- **Rate limiting** — 5-second cooldown between messages
- **Domain restriction** — system prompt limits to schema therapy topics
- **Tool input validation** — Pydantic schemas with max_length on all tool inputs
- **API key management** — all keys in `.env` via python-dotenv

---

## Agent Types (Sprint 3 Concepts)

This project demonstrates several agent architecture concepts:

| Concept | Implementation |
|---|---|
| **ReAct pattern** | LangGraph `create_react_agent` — reason about what tool to use, act, observe, repeat |
| **Goal-based agent** | Agent has the goal of helping with schema therapy; plans tool calls to achieve it |
| **Learning agent** | Feedback loop adapts behaviour based on user ratings |
| **Agentic RAG** | Agent-controlled retrieval — decides when and what to search |
| **Tool state access** | Tools access session state and vector stores independently |

The agent is NOT a simple reflex agent (no memory) or a fixed pipeline (predetermined steps). It dynamically reasons about which tools to use based on the conversation context.

---

## Potential Improvements

- **Human-in-the-loop**: add approval steps before save_session or sensitive recommendations
- **Multi-agent**: separate "diagnosis" and "exercise" agents that collaborate
- **Cloud deployment**: containerise with Docker, deploy to Railway or Render
- **RAGAs evaluation**: objectively measure retrieval quality
- **Fine-tuning**: train a specialised model on schema therapy Q&A pairs
