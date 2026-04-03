# feedback.py  –  Feedback storage and analysis (learning agent pattern)
# Stores user ratings on responses and derives insights that shape future behavior.

import json
from datetime import datetime
from pathlib import Path

FEEDBACK_FILE = Path("data/feedback.json")
FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_feedback() -> list[dict]:
    """Load all stored feedback entries."""
    if not FEEDBACK_FILE.exists():
        return []
    try:
        return json.loads(FEEDBACK_FILE.read_text())
    except Exception:
        return []


def save_feedback_entry(message_index: int, rating: str, response_snippet: str):
    """
    Persist a single feedback entry.
    rating: 'positive' or 'negative'
    """
    entries = load_feedback()
    entries.append({
        "timestamp": datetime.now().isoformat(),
        "message_index": message_index,
        "rating": rating,
        "response_snippet": response_snippet[:300],
    })
    # Rolling window — keep last 200 entries
    entries = entries[-200:]
    FEEDBACK_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2))


def get_feedback_insights() -> str:
    """
    Analyse recent feedback and return insights for the system prompt.
    This is what makes the agent a *learning agent* — it adapts its style
    based on accumulated user ratings.
    """
    entries = load_feedback()
    if len(entries) < 3:
        return ""

    recent = entries[-30:]
    positive = [e for e in recent if e["rating"] == "positive"]
    negative = [e for e in recent if e["rating"] == "negative"]

    if not recent:
        return ""

    insights = []
    pos_rate = len(positive) / len(recent)

    # Overall satisfaction signal
    if pos_rate >= 0.75:
        insights.append(
            "Recent responses have been well-received — continue this approach."
        )
    elif pos_rate <= 0.30:
        insights.append(
            "Recent responses haven't resonated — try being more specific and concise."
        )

    # Patterns in liked responses
    if positive:
        pos_text = " ".join(e["response_snippet"] for e in positive[-5:]).lower()
        if any(w in pos_text for w in ["exercise", "technique", "step", "practice", "try this"]):
            insights.append("The user responds well to practical exercises and techniques.")
        if any(w in pos_text for w in ["schema", "mode", "child", "parent", "protector"]):
            insights.append("The user appreciates schema and mode identification.")
        if any(w in pos_text for w in ["book", "page", "young", "according"]):
            insights.append("The user values book-grounded responses with citations.")

    # Patterns in disliked responses
    if negative:
        avg_len = sum(len(e["response_snippet"]) for e in negative[-5:]) / max(len(negative[-5:]), 1)
        if avg_len > 200:
            insights.append("Disliked responses tend to be long — try being more concise.")

    if not insights:
        return ""

    return "FEEDBACK INSIGHTS (adapt your style accordingly):\n- " + "\n- ".join(insights)
