from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path(".")
data_path = base / "outputs/processed/customer_ml_dataset.csv"
reports = base / "outputs/reports"
figures = base / "outputs/figures"
reports.mkdir(parents=True, exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)

target = "repeat_purchase_next_90d"
if not data_path.exists():
    raise FileNotFoundError(f"Dataset missing: {data_path}")

df = pd.read_csv(data_path)
if target not in df.columns:
    raise ValueError(f"Target column missing: {target}")

features = [
    "total_spend", "total_quantity", "unique_products",
    "unique_invoices", "mean_line_price", "active_purchase_days",
    "recency_days_at_day_90", "average_basket_value",
    "maximum_basket_value", "items_per_invoice",
    "average_days_between_invoices"
]
features = [c for c in features if c in df.columns]
df = df[df[target].isin([0, 1])].copy()

# Group summaries
groups = df.groupby(target)
summary = groups[features].agg(["median", "mean"]).T
summary.to_csv(reports / "repeat_vs_nonrepeat_comparison.csv")

rows = []
for f in features:
    a = pd.to_numeric(df.loc[df[target] == 0, f], errors="coerce").dropna()
    b = pd.to_numeric(df.loc[df[target] == 1, f], errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        continue
    m0, m1 = a.median(), b.median()
    pct = ((m1 - m0) / abs(m0) * 100) if m0 != 0 else np.nan
    rows.append({
        "feature": f,
        "non_repeat_median": m0,
        "repeat_median": m1,
        "repeat_minus_nonrepeat": m1 - m0,
        "median_difference_percent": pct,
        "non_repeat_mean": a.mean(),
        "repeat_mean": b.mean(),
    })

comparison = pd.DataFrame(rows)
comparison["absolute_median_difference"] = comparison["repeat_minus_nonrepeat"].abs()
comparison = comparison.sort_values("absolute_median_difference", ascending=False)
comparison.to_csv(reports / "repeat_purchase_pattern_ranking.csv", index=False)

# Customer group rates
counts = df[target].value_counts().sort_index()
total = len(df)
rate0 = counts.get(0, 0) / total * 100
rate1 = counts.get(1, 0) / total * 100

# Actual pattern chart
if not comparison.empty:
    chart = comparison.head(8).sort_values("repeat_minus_nonrepeat")
    ax = chart.set_index("feature")[
        ["non_repeat_median", "repeat_median"]
    ].plot(kind="barh", figsize=(10, 6))
    ax.set_title("Median Behaviour: Repeat vs Non-repeat Customers")
    ax.set_xlabel("Median feature value")
    plt.tight_layout()
    plt.savefig(figures / "repeat_vs_nonrepeat_patterns.png", dpi=160)
    plt.close()

# Evidence-led narrative; no unsupported causal claims
lines = [
    "# Customer Repeat-Purchase Behaviour: Findings and Recommendations",
    "",
    "## Group definition",
    f"- Non-repeat / target 0: {counts.get(0, 0):,} customers ({rate0:.2f}% of this dataset).",
    f"- Repeat / target 1: {counts.get(1, 0):,} customers ({rate1:.2f}% of this dataset).",
    "- These labels refer to the saved target repeat_purchase_next_90d.",
    "",
    "## What distinguishes the groups?",
    "The table below compares the actual median feature values. Differences are associations, not proof of customer motivation.",
    "",
    "| Behaviour feature | Non-repeat median | Repeat median | Difference |",
    "|---|---:|---:|---:|",
]
for _, r in comparison.head(10).iterrows():
    lines.append(
        f"| {r['feature']} | {r['non_repeat_median']:.3f} | "
        f"{r['repeat_median']:.3f} | {r['repeat_minus_nonrepeat']:.3f} |"
    )

lines += [
    "",
    "## Interpretation for the client",
    "",
    "Use the measured group differences above to identify behaviours associated with repeat purchasing:",
    "- If repeat customers have more invoices or active purchase days, test loyalty rewards and convenient reorder reminders.",
    "- If they buy more unique products, test relevant cross-sell recommendations based on prior purchases.",
    "- If recency differs, test carefully timed re-engagement messages for customers who have become inactive.",
    "- If average order gaps differ, test reminders near customers' observed reorder intervals.",
    "- If spend or basket value differs, test relevant bundles or thresholds; do not assume discounts are necessary.",
    "",
    "## Why some customers may not repeat",
    "This transaction dataset can identify lower purchase frequency, fewer observed products, longer recency, or other measured differences when the table supports them. It cannot establish that customers were unhappy, found prices too high, or experienced delivery problems because those causes are not directly recorded.",
    "",
    "## Recommended next steps",
    "1. Prioritise customers whose behaviour resembles the measured repeat-customer pattern.",
    "2. Run a small, controlled test of reorder reminders, relevant recommendations, or loyalty offers.",
    "3. Compare repeat-purchase rate and revenue between test and comparison groups.",
    "4. Collect optional feedback on price, product fit, and service to investigate the actual reasons for not returning.",
    "",
    "## Files",
    "- Detailed group comparison: `outputs/reports/repeat_vs_nonrepeat_comparison.csv`",
    "- Ranked behavioural differences: `outputs/reports/repeat_purchase_pattern_ranking.csv`",
    "- Chart: `outputs/figures/repeat_vs_nonrepeat_patterns.png`",
]
report = reports / "customer_repeat_purchase_insights.md"
report.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:18]))
print(f"\nSaved report: {report}")
print("Saved group comparison and ranked patterns in outputs/reports/")
print("Saved chart in outputs/figures/")

# Add report and chart to README
readme = base / "README.md"
old = readme.read_text(encoding="utf-8", errors="replace") if readme.exists() else "# Online Retail II\n"
marker = "<!-- CUSTOMER INSIGHTS START -->"
if marker in old:
    old = old.split(marker)[0].rstrip()
old += (
    "\n\n" + marker + "\n\n"
    "## Customer Repeat-Purchase Insights\n\n"
    "The group comparison and ranked behavioural differences are calculated from the processed customer dataset. "
    "Read `outputs/reports/customer_repeat_purchase_insights.md` for the results and business recommendations. "
    "The analysis reports associations rather than unsupported causal claims.\n\n"
    "- Group comparison: `outputs/reports/repeat_vs_nonrepeat_comparison.csv`\n"
    "- Ranked differences: `outputs/reports/repeat_purchase_pattern_ranking.csv`\n"
    "- Plot: `outputs/figures/repeat_vs_nonrepeat_patterns.png`\n"
)
readme.write_text(old, encoding="utf-8")

subprocess.run(["git", "add", "README.md", str(reports), str(figures)], check=True)
subprocess.run(["git", "commit", "-m", "Add evidence-based repeat customer insights"], check=True)
subprocess.run(["git", "push", "origin", "main"], check=True)
print("\nSUCCESS: Customer insights and plots pushed to GitHub.")
