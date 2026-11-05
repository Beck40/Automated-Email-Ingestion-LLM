"""Provider-neutral email field extraction with deterministic fallback."""
import json

from typing import Any


def extract_email(text: str, llm: Any = None) -> dict[str, Any]:
    if llm is not None:
        response = llm.invoke(f"Extract customer, intent, urgency and entities as JSON:\n{text}")
        return json.loads(response.content if hasattr(response, "content") else str(response))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {"customer": lines[0] if lines else "", "intent": "unknown", "urgency": "unknown", "entities": []}
