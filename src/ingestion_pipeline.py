"""Multi-file underwriting submission grouping and batch orchestration."""
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .llm_extractor import extract_submission_data


def group_submission_files(directory: str | Path, submission_ids: Iterable[str] | None = None) -> dict[str, list[Path]]:
    submissions_dir = Path(directory)
    files = sorted(path for path in submissions_dir.glob("*.md") if path.is_file())
    if not files:
        return {}

    grouped_files: dict[str, list[Path]] = {}
    for path in files:
        match = re.match(r"^(E\d{3})", path.name, re.IGNORECASE)
        if not match:
            continue
        submission_id = match.group(1).upper()
        grouped_files.setdefault(submission_id, []).append(path)

    if submission_ids is not None:
        selected = {str(value).upper() for value in submission_ids}
        missing = selected - set(grouped_files)
        if missing:
            raise ValueError(f"Unknown submission IDs: {', '.join(sorted(missing))}")
        grouped_files = {key: grouped_files[key] for key in sorted(selected)}
    return dict(sorted(grouped_files.items()))


def run_batch(directory: str | Path, submission_ids: Iterable[str] | None = None, rate_limit_pause: float = 0.0,
              client: Any | None = None, csv_output: str | Path | None = None,
              json_output: str | Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    grouped_files = group_submission_files(directory, submission_ids)
    for index, (submission_id, submission_files) in enumerate(grouped_files.items()):
        source_text = "\n\n".join(
            f"--- {path.name} ---\n{path.read_text(encoding='utf-8', errors='replace')}"
            for path in submission_files
        )
        records.append(extract_submission_data(submission_id, source_text, client=client))
        if rate_limit_pause > 0 and index + 1 < len(grouped_files):
            time.sleep(rate_limit_pause)

    columns = [
        "submission_id", "company_name", "revenue", "countries", "industry",
        "requested_coverages", "confidence", "review_required",
        "review_reasons", "explanation",
    ]
    if csv_output:
        csv_path = Path(csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).reindex(columns=columns).to_csv(csv_path, index=False)
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract underwriting submissions")
    parser.add_argument("--data-dir", default="data/sample emails")
    parser.add_argument("--submission-ids", nargs="*")
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--csv-output", default="data/llm_extracted_submissions.csv")
    parser.add_argument("--json-output")
    args = parser.parse_args()
    run_batch(args.data_dir, submission_ids=args.submission_ids, rate_limit_pause=args.pause,
              csv_output=args.csv_output, json_output=args.json_output)


if __name__ == "__main__":
    main()

