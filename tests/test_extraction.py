import json
from types import SimpleNamespace

import pandas as pd

from src.evaluate import (
    evaluate_predictions,
    exact_match_rate,
    parse_set,
    semantic_match,
    set_scores,
)
from src.ingestion_pipeline import group_submission_files
from src.llm_extractor import apply_confidence_rules, extract_submission_data


def test_group_submission_files_by_prefix(tmp_path):
    (tmp_path / "E001_email.md").write_text("email", encoding="utf-8")
    (tmp_path / "E001_attachment_1.md").write_text("attachment", encoding="utf-8")
    (tmp_path / "E002_email.md").write_text("email", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    groups = group_submission_files(tmp_path)
    assert list(groups) == ["E001", "E002"]
    assert len(groups["E001"]) == 2
    assert group_submission_files(tmp_path, ["E002"]).keys() == {"E002"}


def test_confidence_rules_only_downgrade():
    data = {"confidence": "High", "review_reasons": [], "review_required": False}
    result = apply_confidence_rules(
        data,
        "The estimated revenue differs from the audited actual revenue.",
    )
    assert result["confidence"] == "Low"
    assert result["review_required"] is True


def test_e047_conflict_reason_caps_high_and_forces_review():
    data = {
        "confidence": "High",
        "review_reasons": [
            "Conflicting company name across the email and change-of-name certificate."
        ],
        "review_required": False,
    }
    result = apply_confidence_rules(data, "The change of name certificate differs.")
    assert result["confidence"] == "Medium"
    assert result["review_required"] is True


class MockCompletions:
    def __init__(self, content):
        self.content = content

    def create(self, **kwargs):
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


class MockClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=MockCompletions(content))


def test_extract_submission_data_parses_mock_json():
    payload = {
        "company_name": "Example Ltd",
        "revenue": 123456,
        "countries": ["United Kingdom"],
        "industry": "Technology",
        "requested_coverages": ["Cyber"],
        "confidence": "High",
        "review_reasons": [],
        "review_required": False,
        "explanation": "Consistent sources.",
    }
    client = MockClient(json.dumps(payload))
    result = extract_submission_data("E001", "Consistent application.", client=client)
    assert result["submission_id"] == "E001"
    assert result["company_name"] == "Example Ltd"
    assert result["revenue"] == 123456


def test_extract_submission_data_malformed_output_fallback():
    client = MockClient("not valid json")
    result = extract_submission_data("E001", "Source text", client=client)
    assert result["confidence"] == "Low"
    assert result["review_required"] is True
    assert result["review_reasons"]


def test_exact_and_semantic_evaluation():
    truth = [{"company_name": "Example Ltd", "revenue": 100}]
    pred = [{"company_name": "Example Ltd", "revenue": 100}]
    assert exact_match_rate(truth, pred, ["company_name", "revenue"]) == {
        "company_name": 1.0,
        "revenue": 1.0,
    }
    assert semantic_match("Retail", "Retail Services")


def test_set_based_precision_recall_f1():
    precision, recall, f1 = set_scores(["UK", "US"], ["UK", "France"])
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5
    assert parse_set("not a list") == set()


def test_evaluate_predictions_and_malformed_isolation(tmp_path):
    ground_truth = pd.DataFrame([
        {
            "submission_id": "E001", "company_name": "Example Ltd", "revenue": 100,
            "countries": "['UK']", "industry": "Technology",
            "requested_coverages": "['Cyber']",
        }
    ])
    predictions = pd.DataFrame([
        {
            "submission_id": "E001", "company_name": "Example Ltd", "revenue": 101,
            "countries": "['UK']", "industry": "Tech",
            "requested_coverages": "['Cyber']",
        }
    ])
    truth_path = tmp_path / "truth.csv"
    pred_path = tmp_path / "predictions.csv"
    ground_truth.to_csv(truth_path, index=False)
    predictions.to_csv(pred_path, index=False)

    metrics, errors = evaluate_predictions(truth_path, pred_path, tmp_path / "out")
    assert not metrics.empty
    assert len(errors) == 1
    malformed = metrics.loc[
        metrics["metric"] == "Malformed / failed rate", "score"
    ].iloc[0]
    assert malformed == 0.0
