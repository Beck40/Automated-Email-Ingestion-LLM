# Automated Email Ingestion LLM

## Objective

Manual triage of operational email is slow, inconsistent and difficult to audit. This project extracts customer, intent, urgency and entities from unstructured messages, creating a measurable interface between language models and downstream service operations. It is designed as a personal engineering project exploring how LLM automation can be introduced without weakening control over uncertain cases.

## Architecture

```text
Automated-Email-Ingestion-LLM/
├── assets/
│   └── charts/                         # Accuracy and fallback-risk visuals
├── data/                               # Sample emails and evaluation outputs
├── docs/
│   └── Email_Ingestion_Methodology.md
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_email_ingestion.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   ├── ingestion_pipeline.py           # Batch orchestration
│   ├── llm_extractor.py                # Provider boundary and fallback
│   ├── evaluate.py                     # Field-level evaluation
│   └── generate_charts.py              # Governance charts
└── tests/
```

## Quickstart

```bash
pip install -r requirements.txt
python -c "from src.ingestion_pipeline import process; process('data/sample emails')"
python -m src.generate_charts
```

The local sample emails and evaluation outputs are retained under `data/` for reproducible analysis.

![Extraction accuracy](assets/charts/01_extraction_accuracy.png)

Field-level accuracy separates extraction tasks that can support straight-through processing from fields that need a review gate. This provides a governance baseline for confidence thresholds, model-version comparisons and controlled release decisions.

![Edge-case error rates](assets/charts/02_edge_case_error_rates.png)

Edge-case error rates show where the Model Governance & Fallback design must intervene. Ambiguous or low-confidence messages can be routed to human review, preserving operational control while allowing high-confidence cases to flow automatically.
