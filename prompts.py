# prompts.py  –  Load externalized prompts from YAML

from pathlib import Path
import yaml

PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"


def load_prompts() -> dict:
    """Load all prompts from prompts.yaml."""
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
