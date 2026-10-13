from pathlib import Path
import json
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(".")
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"
PROCESSED = ROOT / "outputs" / "processed"

REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

def find_col(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    for c in df.columns:
        if any(term in str(c).lower() for term in candidates):
            return c
    return None

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=160, bbox_inches="tight")
    plt.close()

print("\n=== 1. REPEAT CUSTOMER ANALYSIS ===")

path = PROCESSED / "cleaned_transactions.csv"
if not path.exists():
    raise FileNotFoundError(f"Dataset not found: {path}")

df = pd.read_csv(path, low_memory=False)
print("Cleaned transaction rows:", len(df))
print("Dataset columns:", list(df.columns))

customer = find_col(df, ["Customer ID", "CustomerID", "customer_id"])
invoice = find_col(df, ["Invoice", "InvoiceNo", "Invoice Number"])
date = find_col(df, ["InvoiceDate", "Invoice Date", "date"])
product = find_col(df, ["StockCode", "Stock Code", "product_code"])

if not all([customer, invoice, date, product]):
    raise ValueError(
        "Could not identify customer/invoice/date/product columns. "
        f"Available columns: {list(df.columns)}"
    )

df[date] = pd.to_datetime(df[date], errors="coerce", dayfirst=True)
df = df.dropna(subset=[customer, invoice, date, product]).copy()
invoice_text = df[invoice].astype(str).str.strip().str.upper()
df = df[~invoice_text.str.startswith("C")].copy()

orders = (
    df.groupby([customer, invoice], as_index=False)
      .agg(order_date=(date, "min"))
      .sort_values([customer, "order_date", invoice])
)
orders["gap_days"] = (
    orders.groupby(customer)["order_date"].diff().dt.total_seconds() / 86400
)

order_counts = orders.groupby(customer)[invoice].nunique()
repeat_ids = order_counts[order_counts >= 2].index

repeat_df = df[df[customer].isin(repeat_ids)].copy()

# For each customer-product pair, count invoices containing that product.
product_orders = (
    repeat_df.groupby([customer, product])[invoice].nunique()
)
repeated_product_counts = (
    product_orders[product_orders >= 2].groupby(level=0).size()
)

customer_report = pd.DataFrame({
    "total_orders": order_counts.loc[repeat_ids],
    "unique_products": repeat_df.groupby(customer)[product].nunique()
})
customer_report["products_bought_again"] = (
    repeated_product_counts.reindex(customer_report.index).fillna(0).astype(int)
)
customer_report["product_pattern"] = customer_report[
    "products_bought_again"
].gt(0).map({
    True: "Repeated at least one product",
    False: "No product repeated across invoices"
})

gaps = orders[orders[customer].isin(repeat_ids)].groupby(customer)["gap_days"]
gap_stats = gaps.agg(
    order_gaps_count="count",
    average_gap_days="mean",
    median_gap_days="median"
)
customer_report = customer_report.join(gap_stats)
customer_report.index.name = "Customer_ID"
customer_report.to_csv(REPORTS / "repeat_customer_behavior.csv")

n_repeat = len(customer_report)
n_same = int(customer_report["products_bought_again"].gt(0).sum())
n_different = n_repeat - n_same

valid_gaps = orders.loc[
    orders[customer].isin(repeat_ids), "gap_days"
].dropna()

summary_rows = [
    ("Customers with at least 2 invoices", n_repeat),
    ("Customers who repeated at least one product", n_same),
    ("Customers with no product repeated across invoices", n_different),
    ("Percent repeating at least one product",
     round(100 * n_same / n_repeat, 2) if n_repeat else 0),
    ("Mean days between consecutive invoices", round(valid_gaps.mean(), 2)
     if len(valid_gaps) else None),
    ("Median days between consecutive invoices", round(valid_gaps.median(), 2)
     if len(valid_gaps) else None),
]
summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
summary.to_csv(REPORTS / "repeat_customer_behavior_summary.csv", index=False)
print(summary.to_string(index=False))

# Plot 1: repeated product behavior
plt.figure(figsize=(8, 5))
plt.bar(
    ["Repeated a product", "No repeated product"],
    [n_same, n_different]
)
plt.title("Repeat Customers: Product Repetition")
plt.ylabel("Number of customers")
save_plot("repeat_customer_product_patterns.png")

# Plot 2: order-gap distribution
if len(valid_gaps):
    plt.figure(figsize=(8, 5))
    plt.hist(valid_gaps.clip(upper=valid_gaps.quantile(0.99)),
             bins=30, edgecolor="black")
    plt.title("Days Between Consecutive Customer Orders")
    plt.xlabel("Days between invoices (upper 1% clipped for display)")
    plt.ylabel("Number of order gaps")
    save_plot("repeat_customer_order_gap_distribution.png")

# Plot 3: most commonly repeated products across invoices
if len(product_orders):
    repeated_codes = (
        product_orders[product_orders >= 2]
        .groupby(level=1).size()
        .sort_values(ascending=False).head(15).sort_values()
    )
    if len(repeated_codes):
        plt.figure(figsize=(9, 6))
        plt.barh(repeated_codes.index.astype(str), repeated_codes.values)
        plt.title("Top Products Repeated Across Customer Invoices")
        plt.xlabel("Customers who purchased product on 2+ invoices")
        save_plot("most_repeated_products.png")

print("\n=== 2. MACHINE LEARNING RESULTS ===")

# Print saved project summary if available.
summary_path = REPORTS / "project_summary.json"
if summary_path.exists():
    project_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("\nProject summary:")
    print(json.dumps(project_summary, indent=2, default=str))

# Show actual model comparison metrics from the generated report.
model_path = REPORTS / "model_comparison.csv"
if model_path.exists():
    models = pd.read_csv(model_path)
    print("\nModel comparison columns:", list(models.columns))
    print(models.to_string(index=False))

    # Plot available performance metrics.
    model_col = find_col(models, ["model", "model_name", "classifier", "name"])
    for metric in ["f1", "f1_score", "roc_auc", "pr_auc", "accuracy",
                   "precision", "recall"]:
        metric_col = find_col(models, [metric])
        if model_col and metric_col and pd.api.types.is_numeric_dtype(models[metric_col]):
            plot_df = models[[model_col, metric_col]].dropna()
            if len(plot_df):
                plot_df = plot_df.sort_values(metric_col)
                plt.figure(figsize=(10, max(5, len(plot_df) * 0.35)))
                plt.barh(plot_df[model_col].astype(str), plot_df[metric_col])
                plt.title(f"Model Comparison: {metric_col}")
                plt.xlabel(metric_col)
                save_plot(f"model_comparison_{metric_col}.png")

# Print classification report if available.
class_report = REPORTS / "best_model_classification_report.txt"
if class_report.exists():
    print("\nBest model classification report:")
    print(class_report.read_text(encoding="utf-8"))

# Show target class balance and calculate imbalance ratio if report exists.
target_path = REPORTS / "target_class_distribution.csv"
if target_path.exists():
    target_df = pd.read_csv(target_path)
    print("\nTarget class distribution:")
    print(target_df.to_string(index=False))
    numeric_cols = [
        c for c in target_df.columns
        if pd.api.types.is_numeric_dtype(target_df[c])
    ]
    if numeric_cols and len(target_df):
        counts = target_df[numeric_cols[-1]].dropna()
        if len(counts) >= 2 and counts.min() > 0:
            print("Class imbalance ratio (majority:minority):",
                  f"{counts.max() / counts.min():.2f}:1")

# Print feature names from the customer ML dataset.
ml_path = PROCESSED / "customer_ml_dataset.csv"
if ml_path.exists():
    ml_df = pd.read_csv(ml_path, nrows=5)
    print("\nML dataset columns:", list(ml_df.columns))
    target_col = find_col(
        ml_df,
        ["target", "label", "repeat_purchase", "will_repeat",
         "repeat_purchase_target", "target_repeat_purchase"]
    )
    if target_col:
        feature_cols = [c for c in ml_df.columns if c != target_col]
        print("Detected target column:", target_col)
        print("Candidate feature names:", feature_cols)
        print("Candidate feature count:", len(feature_cols))
        print("Note: verify feature list against run_project.py; "
              "IDs or leakage columns must not be counted as model inputs.")
    else:
        print("Target column was not identified automatically. "
              "Use run_project.py and project_summary.json to verify "
              "the exact feature columns.")

print("\n=== COMPLETED ===")
print("Reports saved in:", REPORTS.resolve())
print("Plots saved in:", FIGURES.resolve())
