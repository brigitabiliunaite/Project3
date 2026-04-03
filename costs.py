# costs.py  –  Token usage and cost tracking (multi-model support)

# Pricing per 1K tokens, USD — updated for multi-model support
PRICING = {
    # OpenAI
    "gpt-4o-mini":               {"input": 0.000150, "output": 0.000600},
    "gpt-4o":                    {"input": 0.002500, "output": 0.010000},
    # Anthropic
    "claude-sonnet-4-20250514":  {"input": 0.003000, "output": 0.015000},
    "claude-haiku-4-20250414":   {"input": 0.000800, "output": 0.004000},
    # Embeddings
    "text-embedding-3-small":    {"input": 0.000020, "output": 0.0},
}

# Friendly display names for the UI
MODEL_DISPLAY_NAMES = {
    "gpt-4o-mini":               "GPT-4o Mini",
    "gpt-4o":                    "GPT-4o",
    "claude-sonnet-4-20250514":  "Claude Sonnet 4",
    "claude-haiku-4-20250414":   "Claude Haiku 4",
}

# Models grouped by provider
PROVIDER_MODELS = {
    "OpenAI":    ["gpt-4o-mini", "gpt-4o"],
    "Anthropic": ["claude-sonnet-4-20250514", "claude-haiku-4-20250414"],
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return cost in USD for a given model call."""
    p = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens / 1000) * p["input"] + (output_tokens / 1000) * p["output"]


def format_cost(usd: float) -> str:
    """Format a USD amount nicely."""
    if usd < 0.001:
        return f"${usd:.6f}"
    return f"${usd:.4f}"
