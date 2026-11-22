"""Evaluation metrics for structured underwriting extraction."""
from ast import literal_eval
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def parse_set(value: Any) -> set[Any]:
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    if not isinstance(value, str) or not value.strip():
        return set()
    try:
        parsed = literal_eval(value)
    except (SyntaxError, ValueError):
        return set()
    if not isinstance(parsed, (list, tuple, set)):
        return set()
    return {str(item).strip() for item in parsed if str(item).strip()}


def set_scores(predicted: Any, actual: Any) -> tuple[float, float, float]:
    predicted_set = parse_set(predicted)
    actual_set = parse_set(actual)
    true_positives = len(predicted_set & actual_set)
    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(actual_set) if actual_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def exact_match_rate(expected: Iterable[Mapping[str, Any]], predicted: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> dict[str, float]:
    expected_list = list(expected)
    predicted_list = list(predicted)
    total = len(expected_list)
    if total == 0:
        return {field: 0.0 for field in fields}
    return {
        field: sum(
            (truth_row.get(field) == pred_row.get(field))
            for truth_row, pred_row in zip(expected_list, predicted_list)
        ) / total
        for field in fields
    }


def semantic_match(predicted: Any, actual: Any) -> bool:
    if not isinstance(predicted, str) or not isinstance(actual, str):
        return False
    predicted_lower = predicted.strip().casefold()
    actual_lower = actual.strip().casefold()
    return bool(predicted_lower and actual_lower and (
        predicted_lower in actual_lower or actual_lower in predicted_lower
    ))


def malformed_or_failed(row: Mapping[str, Any]) -> bool:
    required_scalars = ("company_name_pred", "revenue_pred", "industry_pred")
    if not row.get("record_present", False):
        return True
    if any(pd.isna(row.get(field)) for field in required_scalars):
        return True
    for field in ("countries_pred", "requested_coverages_pred"):
        value = row.get(field)
        if isinstance(value, str):
            try:
                parsed = literal_eval(value)
            except (SyntaxError, ValueError):
                return True
        else:
            parsed = value
        if not isinstance(parsed, (list, tuple, set)):
            return True
    return False


def evaluate_predictions(ground_truth_path: str | Path, predictions_path: str | Path, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ground_truth = pd.read_csv(ground_truth_path)
    predictions = pd.read_csv(predictions_path)

    comparison = ground_truth.merge(
        predictions,
        on="submission_id",
        how="left",
        suffixes=("_truth", "_pred"),
        indicator=True,
    )
    comparison["record_present"] = comparison["_merge"] == "both"

    for field in ("countries", "requested_coverages"):
        score_columns = [f"{field}_precision", f"{field}_recall", f"{field}_f1"]
        comparison[score_columns] = comparison.apply(
            lambda row: set_scores(row[f"{field}_pred"], row[f"{field}_truth"]),
            axis=1,
            result_type="expand",
        )
    comparison["company_name_match"] = comparison["company_name_pred"] == comparison["company_name_truth"]
    comparison["revenue_match"] = comparison["revenue_pred"] == comparison["revenue_truth"]
    comparison["industry_match"] = comparison.apply(
        lambda row: semantic_match(row["industry_pred"], row["industry_truth"]),
        axis=1,
    )
    comparison["malformed_or_failed"] = comparison.apply(malformed_or_failed, axis=1)

    metrics = pd.DataFrame([
        ["company_name", "Exact match accuracy (strict)", comparison["company_name_match"].mean()],
        ["revenue", "Exact match accuracy (strict)", comparison["revenue_match"].mean()],
        ["industry", "Semantic match accuracy", comparison["industry_match"].mean()],
        ["countries", "Macro precision", comparison["countries_precision"].mean()],
        ["countries", "Macro recall", comparison["countries_recall"].mean()],
        ["countries", "Macro F1", comparison["countries_f1"].mean()],
        ["requested_coverages", "Macro precision", comparison["requested_coverages_precision"].mean()],
        ["requested_coverages", "Macro recall", comparison["requested_coverages_recall"].mean()],
        ["requested_coverages", "Macro F1", comparison["requested_coverages_f1"].mean()],
        ["record", "Malformed / failed rate", comparison["malformed_or_failed"].mean()],
    ], columns=["field", "metric", "score"])
    metrics["score_percent"] = (metrics["score"] * 100).round(2)

    error_columns = [
        "submission_id", "malformed_or_failed", "company_name_truth", "company_name_pred",
        "revenue_truth", "revenue_pred", "countries_truth", "countries_pred",
        "industry_truth", "industry_pred", "requested_coverages_truth",
        "requested_coverages_pred", "confidence", "explanation",
    ]
    errors = comparison.loc[
        comparison["malformed_or_failed"]
        | ~comparison["company_name_match"]
        | ~comparison["revenue_match"]
        | ~comparison["industry_match"]
        | (comparison["countries_f1"] < 1)
        | (comparison["requested_coverages_f1"] < 1),
        [column for column in error_columns if column in comparison.columns],
    ]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "evaluation_metrics.csv", index=False)
    comparison.to_csv(output / "evaluation_row_level.csv", index=False)
    errors.to_csv(output / "evaluation_errors.csv", index=False)
    return metrics, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate underwriting extraction predictions")
    parser.add_argument("--ground-truth", default="data/ground_truth.csv")
    parser.add_argument("--predictions", default="data/llm_extracted_submissions.csv")
    parser.add_argument("--output-dir", default="data/evaluation_outputs")
    args = parser.parse_args()
    evaluate_predictions(args.ground_truth, args.predictions, args.output_dir)

