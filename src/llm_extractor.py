"""LLM-backed underwriting extraction and deterministic review rules."""
import json
from typing import Any

from groq import Groq

from .config import get_settings


CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}


def apply_confidence_rules(data: dict[str, Any], source_text: str) -> dict[str, Any]:
    text = source_text.casefold()
    low_reasons: list[str] = []
    medium_reasons: list[str] = []

    if (("standalone" in text or "legal entity" in text)
            and ("group" in text or "consolidated" in text)):
        low_reasons.append("Group and standalone information are both present.")
    if (("estimate" in text or "estimated" in text)
            and ("audited" in text or "actual" in text)):
        low_reasons.append("Estimated and audited/actual figures are both present.")
    if "expiring" in text and "requested" in text:
        medium_reasons.append("Expiring and newly requested cover are both mentioned.")
    if (("primary address" in text or "registered office" in text)
            and ("operating" in text or "territor" in text)):
        medium_reasons.append("Address and operating-territory countries may differ.")

    model_reasons = data.get("review_reasons", [])
    if not isinstance(model_reasons, list):
        model_reasons = []
    reasons = list(dict.fromkeys(
        [str(reason) for reason in model_reasons] + low_reasons + medium_reasons
    ))
    data["review_reasons"] = reasons

    llm_confidence = str(data.get("confidence", "Low")).title()
    if llm_confidence not in CONFIDENCE_RANK:
        llm_confidence = "Low"
    rule_confidence = "Low" if low_reasons else "Medium" if medium_reasons else "High"

    final_confidence = min(
        (llm_confidence, rule_confidence),
        key=lambda value: CONFIDENCE_RANK[value],
    )

    if reasons:
        final_confidence = min(
            (final_confidence, "Medium"),
            key=lambda value: CONFIDENCE_RANK[value],
        )
        data["review_required"] = True
    else:
        data["review_required"] = final_confidence != "High"

    data["confidence"] = final_confidence
    data.setdefault("submission_id", None)
    data.setdefault("company_name", None)
    data.setdefault("revenue", None)
    data.setdefault("countries", [])
    data.setdefault("industry", None)
    data.setdefault("requested_coverages", [])
    data.setdefault("explanation", "")
    return data


EXTRACTION_PROMPT = """Extract the underwriting fields from the email and attachments. Return only valid JSON.
Compare every document before choosing a value. Confidence is based on source agreement, not ground truth.
Use High only when every core field is explicit and consistent. Use Medium for a plausible alternative interpretation.
Use Low when documents conflict or the correct basis cannot be determined.
Use last completed actual revenue, not projected revenue. Distinguish standalone/group, audited/estimated,
operating countries/domicile, and requested/expiring cover. Do not invent missing values.
Return exactly this JSON schema:
{"submission_id":"string","company_name":"string or null","revenue":integer or null,
"countries":[],"industry":"string or null","requested_coverages":[],"confidence":"High|Medium|Low",
"review_required":true,"review_reasons":[],"explanation":"brief evidence-based explanation"}"""


def extract_submission_data(
    submission_id: str,
    source_text: str,
    client: Any | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    if client is None:
        if settings.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY is required for live Groq extraction")
        client = Groq(api_key=settings.groq_api_key.get_secret_value())

    try:
        response = client.chat.completions.create(
            model=settings.groq_model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract this submission:\n{source_text}"},
            ],
        )
        raw_output = response.choices[0].message.content.strip()
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_output)
        if not isinstance(data, dict):
            raise ValueError("LLM output is not a JSON object")
        data["submission_id"] = submission_id
        return apply_confidence_rules(data, source_text)
    except Exception as error:
        return {
            "submission_id": submission_id,
            "company_name": None,
            "revenue": None,
            "countries": [],
            "industry": None,
            "requested_coverages": [],
            "confidence": "Low",
            "review_required": True,
            "review_reasons": [f"Extraction failed: {error}"],
            "explanation": "The submission needs manual review.",
        }
