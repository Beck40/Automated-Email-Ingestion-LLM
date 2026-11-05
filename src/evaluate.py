"""Evaluation metrics for structured email extraction."""
from collections.abc import Iterable


def exact_match_rate(expected: Iterable[dict], predicted: Iterable[dict], fields: Iterable[str]) -> dict[str, float]:
    expected, predicted = list(expected), list(predicted)
    return {field: sum(a.get(field) == b.get(field) for a,b in zip(expected,predicted)) / max(1,len(expected)) for field in fields}


def edge_case_error_rate(expected, predicted, field="intent"):
    return 1 - exact_match_rate(expected, predicted, [field])[field]

