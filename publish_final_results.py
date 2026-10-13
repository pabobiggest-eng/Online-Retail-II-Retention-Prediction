import json
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(".")
reports = root / "outputs" / "reports"
figures = root / "outputs" / "figures"
reports.mkdir(parents=True, exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)

target = "repeat_purchase_next_90d"
data_path = root / "outputs" / "processed" / "customer_ml_dataset.csv"
report_path = reports / "best_model_classification_report.txt"
summary_path = reports / "project_summary.json"
comparison_path = reports / "model_comparison.csv"
repeat_path = reports / "repeat_customer_behavior_summary.csv"

# Read actual model evaluation
if not report_path.exists():
    raise FileNotFoundError(f"Missing: {report_path}")

report_text = report_path.read_text(encoding="utf-8", errors="replace")
summary = {}
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

# Actual features: read the pipeline source and identify the training feature list.
source = (root / "run_project.py").read_text(encoding="utf-8", errors="replace")
feature_count = summary.get("feature_count", "See pipeline")
feature_note = (
    "Feature count reported by the training pipeline: "
    + str(feature_count)
    + ". See run_project.py for the exact model input selection."
)

# Dataset class ratio (full dataset only; not mislabeled as training ratio)
imbalance_note = "Training-set class counts were not independently recorded."
if data_path.exists():
    df = pd.read_csv(data_path)
    if target in df.columns:
        counts = df[target].value_counts().sort_index()
        if len(counts) == 2 and counts.min() > 0:
            imbalance_note = (
                f"Full customer dataset class counts: {counts.to_dict()}. "
                f"Full-dataset majority/minority ratio: "
                f"{counts.max()/counts.min():.3f}:1. "
                "This is not the training-only ratio."
            )

# Actual repeat-customer comparison results
repeat_text = "Repeat-customer summary CSV not found; no exact figures claimed."
if repeat_path.exists():
    repeat_text = repeat_path.read_text(encoding="utf-8", errors="replace")

# Accuracy, precision, recall, F1 from the actual saved model comparison
model_text = ""
if comparison_path.exists():
    models = pd.read_csv(comparison_path)
    model_text = models.to_markdown(index=False) if hasattr(models, "to_markdown") else models.to_string(index=False)

# Plot actual saved model comparison if columns are available
if comparison_path.exists():
    models = pd.read_csv(comparison_path)
    metric_cols = [
        c for c in models.columns
        if c.lower() in {"accuracy", "precision", "recall", "f1", "f1_score", "f1-score"}
    ]
    name_col = next((c for c in models.columns if "model" in c.lower()), models.columns[0])
    if metric_cols:
        ax = models.set_index(name_col)[metric_cols].plot(kind="bar", figsize=(12, 6))
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(figures / "final_model_metrics.png", dpi=160)
        plt.close()

readme = root / "README.md"
old = readme.read_text(encoding="utf-8", errors="replace") if readme.exists() else "# Online Retail II Analysis\n"
marker = "<!-- FINAL VERIFIED RESULTS START -->"
if marker in old:
    old = old.split(marker)[0].rstrip()

section = f"""
{marker}

## Final Results and Model Evaluation

### Classification report
```text
{report_text.strip()}
```

### Model input features
{feature_note}

Candidate columns are not automatically model inputs. Identifier and split metadata must not be counted as predictors. Verify the actual selected feature list in `run_project.py`.

### Class imbalance
{imbalance_note}

### Model comparison
```text
{model_text if model_text else "Model comparison CSV not found."}
```

### Repeat-customer behaviour and potential drivers
```text
{repeat_text.strip()}
```

Interpretation: purchase frequency, product variety, spending, recency and order gaps are behavioural factors to investigate. They should be called associated factors only when supported by the computed comparison; transaction data alone cannot prove customers' personal reasons.

### Figures
- Model metrics chart: `outputs/figures/final_model_metrics.png`
- Other EDA and repeat-customer plots: `outputs/figures/`

<!-- FINAL VERIFIED RESULTS END -->
"""
readme.write_text(old + "\n\n" + section, encoding="utf-8")

print("README updated:", readme)
print("Plots/reports folder:", figures.resolve())
print("\n--- Saved classification report ---")
print(report_text)

# Commit and push
subprocess.run(["git", "add", "README.md", "outputs/reports", "outputs/figures"], check=True)
subprocess.run(["git", "commit", "-m", "Add final verified results and model analysis"], check=True)
subprocess.run(["git", "push", "origin", "main"], check=True)
print("\nSUCCESS: Final results committed and pushed to GitHub.")
