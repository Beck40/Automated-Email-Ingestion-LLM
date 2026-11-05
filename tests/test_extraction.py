from src.evaluate import exact_match_rate, edge_case_error_rate
from src.llm_extractor import extract_email


def test_empty_email_uses_safe_fallback():
    result = extract_email("")
    assert result["customer"] == ""
    assert result["intent"] == "unknown"
    assert result["entities"] == []


def test_evaluation_handles_empty_inputs_and_edge_cases():
    assert exact_match_rate([], [], ["intent"]) == {"intent": 0.0}
    assert edge_case_error_rate([{"intent": "refund"}], [{"intent": "unknown"}]) == 1.0
