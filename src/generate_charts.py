"""Generate charts from the repository's real evaluation CSV outputs."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_corporate_style() -> None:
    sns.set_theme(style="whitegrid", rc={
        "axes.facecolor": "#F8F9FA",
        "figure.facecolor": "#FFFFFF",
        "axes.edgecolor": "#E0E0E0",
        "grid.color": "#EBEBEB",
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "text.color": "#212529",
    })


def generate(
    output_dir: str | Path = "assets/charts",
    data_dir: str | Path = "data",
) -> None:
    set_corporate_style()
    out = Path(output_dir)
    data = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = data / "evaluation_outputs" / "evaluation_metrics.csv"
    row_path = data / "evaluation_outputs" / "evaluation_row_level.csv"
    metrics = pd.read_csv(metrics_path)
    rows = pd.read_csv(row_path)

    labels = {
        "company_name": "Company Name",
        "revenue": "Revenue",
        "industry": "Industry",
        "countries": "Countries",
        "requested_coverages": "Coverages",
        "record": "Malformed",
    }
    wanted = [
        ("company_name", "Exact match accuracy (strict)"),
        ("revenue", "Exact match accuracy (strict)"),
        ("industry", "Semantic match accuracy"),
        ("countries", "Macro F1"),
        ("requested_coverages", "Macro F1"),
        ("record", "Malformed / failed rate"),
    ]
    values = []
    display_labels = []
    for field, metric in wanted:
        match = metrics[(metrics["field"] == field) & (metrics["metric"] == metric)]
        score = float(match["score"].iloc[0]) if not match.empty else 0.0
        values.append(score * 100)
        display_labels.append(labels[field])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2B5B84", "#336699", "#2A9D8F", "#6C5CE7", "#E76F51", "#C1121F"]
    bars = ax.barh(display_labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)")
    ax.set_title("Field Evaluation Metrics | 50-Submission Ground Truth")
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%",
                va="center", fontweight="bold")
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(out / "01_field_evaluation_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    confidence_counts = rows["confidence"].fillna("Unknown").value_counts().reindex(
        ["High", "Medium", "Low"], fill_value=0
    )
    review_required = int(pd.to_numeric(rows["review_required"], errors="coerce").fillna(False).sum()) if "review_required" in rows else 0
    malformed = int(pd.to_numeric(rows["malformed_or_failed"], errors="coerce").fillna(False).sum()) if "malformed_or_failed" in rows else 0
    discrepancy_count = len(pd.read_csv(data / "evaluation_outputs" / "evaluation_errors.csv")) if (data / "evaluation_outputs" / "evaluation_errors.csv").exists() else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax1.bar(confidence_counts.index, confidence_counts.values,
            color=["#2A9D8F", "#E9C46A", "#E76F51"])
    ax1.set_title("Confidence Tier Distribution")
    ax1.set_ylabel("Submissions")
    ax1.set_ylim(0, max(30, int(confidence_counts.max()) + 4))
    for index, value in enumerate(confidence_counts.values):
        ax1.text(index, value + 0.5, str(int(value)), ha="center", fontweight="bold")

    error_labels = ["Review required", "Field discrepancies", "Malformed"]
    error_values = [review_required, discrepancy_count, malformed]
    bars = ax2.bar(error_labels, error_values,
                   color=["#5B8FF9", "#E76F51", "#C1121F"])
    ax2.set_title("Review and Error Breakdown")
    ax2.set_ylabel("Submissions")
    ax2.tick_params(axis="x", rotation=20)
    ax2.set_ylim(0, max(35, max(error_values) + 5))
    for bar, value in zip(bars, error_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value),
                 ha="center", fontweight="bold")
    fig.suptitle("Confidence and Discrepancy Summary | 50-Submission Evaluation",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "02_confidence_and_error_breakdown.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    for old_name in ("01_extraction_accuracy.png", "02_edge_case_error_rates.png"):
        old_path = out / old_name
        if old_path.exists():
            old_path.unlink()


if __name__ == "__main__":
    generate()

