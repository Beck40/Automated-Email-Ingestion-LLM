"""Batch email ingestion and structured extraction orchestration."""
from pathlib import Path
import json
from .llm_extractor import extract_email


def load_emails(directory: str | Path) -> list[dict[str, str]]:
    root = Path(directory)
    return [{"id": p.stem, "text": p.read_text(encoding="utf-8", errors="replace")} for p in sorted(root.glob("*.txt"))]


def process(directory: str | Path, output: str | Path = "data/extractions.json"):
    results = [extract_email(item["text"]) | {"id": item["id"]} for item in load_emails(directory)]
    Path(output).parent.mkdir(parents=True, exist_ok=True); Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results

