"""STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)

Reads outputs/processed/cleaned_transactions.csv and creates report tables,
figures and outputs/reports/eda_summary.md.

Run from the project root:
    python .\\02_eda.py

This version intentionally does not import retail_utils because that module
is not present in the current project. It uses the cleaned CSV created by
run_project.py.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Create EDA reports and charts for Online Retail II.")
parser.add_argument("--clean", default="outputs/processed/cleaned_transactions.csv")
parser.add_argument("--reports", default="outputs/reports")
parser.add_argument("--figures", default="outputs/figures")
args = parser.parse_args()

REPORTS, FIGS = Path(args.reports), Path(args.figures)
REPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
})
BLUE, ORANGE, GREY = "#2f6db5", "#e8833a", "#9aa5b1"


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def save_table(table, name, index=True):
    table.to_csv(REPORTS / f"{name}.csv", index=index)


def normalize_columns(frame):
    """Map common variants of Online Retail II column names to standard names."""
    key_to_actual = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): col for col in frame.columns
    }
    aliases = {
        "Invoice": ["invoice", "invoiceno", "invoicenumber"],
        "StockCode": ["stockcode", "stock", "productcode"],
        "Description": ["description", "productdescription"],
        "Quantity": ["quantity", "qty"],
        "InvoiceDate": ["invoicedate", "date", "invoicedatetime"],
        "Price": ["price", "unitprice", "unitcost"],
        "Customer ID": ["customerid", "customer", "customeridentifier"],
        "Country": ["country"],
        "Sales": ["sales", "revenue", "linetotal", "amount"],
    }
    rename = {}
    for standard, candidates in aliases.items():
        for candidate in candidates:
            if candidate in key_to_actual:
                rename[key_to_actual[candidate]] = standard
                break
    return frame.rename(columns=rename)


# ======================================================================
# 1. LOAD CLEANED DATA
# ======================================================================
clean_path = Path(args.clean)
if not clean_path.exists():
    raise FileNotFoundError(
        f"Cleaned data not found: {clean_path}\n"
        "Run `python .\\run_project.py` first, then run `python .\\02_eda.py`."
    )

print(f"Loading cleaned file: {clean_path}")
df = pd.read_csv(clean_path, low_memory=False)
df = normalize_columns(df)

required = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(
        f"The cleaned CSV is missing required columns: {missing}. "
        f"Columns found: {list(df.columns)}"
    )

# Parse types safely; the CSV is already cleaned by the main pipeline.
df["Invoice"] = df["Invoice"].astype("string").str.strip()
df["Customer ID"] = df["Customer ID"].astype("string").str.replace(r"\.0$", "", regex=True)
df["StockCode"] = df["StockCode"].astype("string").str.strip()
df["Country"] = df["Country"].astype("string").str.strip()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
for col in ["Quantity", "Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if "Sales" not in df.columns:
    df["Sales"] = df["Quantity"] * df["Price"]
else:
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
if "Description" not in df.columns:
    df["Description"] = "Unknown product"

before = len(df)
df = df.dropna(subset=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Sales"]).copy()
df = df[df["Invoice"].str.len().gt(0) & df["StockCode"].str.len().gt(0)].copy()
print(f"Rows used for EDA: {len(df):,} (invalid required rows skipped: {before - len(df):,})")
if df.empty:
    raise ValueError("No usable rows remain in the cleaned CSV after parsing required columns.")

df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")
last_date = df["InvoiceDate"].max()
partial_month = (
    pd.Period(last_date, freq="M")
    if last_date.day < last_date.days_in_month - 1
    else None
)

# One row per invoice
invoices = (
    df.groupby("Invoice", dropna=False)
    .agg(
        customer=("Customer ID", "first"),
        country=("Country", "first"),
        date=("InvoiceDate", "min"),
        revenue=("Sales", "sum"),
        units=("Quantity", "sum"),
        lines=("StockCode", "count"),
    )
    .reset_index()
)
invoices["month"] = invoices["date"].dt.to_period("M")

# Customer metrics exclude invoices with no identifiable customer ID.
customer_invoices = invoices.dropna(subset=["customer"]).copy()
first_month = customer_invoices.groupby("customer")["month"].min()
customers = (
    customer_invoices.groupby("customer")
    .agg(
        orders=("Invoice", "nunique"),
        revenue=("revenue", "sum"),
        first_order=("date", "min"),
        last_order=("date", "max"),
        country=("country", "first"),
    )
)
customers["is_repeat"] = customers["orders"] >= 2
if customers.empty:
    raise ValueError("No customer IDs were found; customer and cohort analysis cannot be calculated.")

# ======================================================================
# 2. KPIs
# ======================================================================
cust_sorted = customers["revenue"].sort_values(ascending=False)
total_customer_revenue = customers["revenue"].sum()
top20_n = max(1, int(np.ceil(0.2 * len(cust_sorted))))
top20_share = cust_sorted.head(top20_n).sum() / total_customer_revenue if total_customer_revenue else np.nan
total_sales = df["Sales"].sum()
uk_sales = df.loc[df["Country"].eq("United Kingdom"), "Sales"].sum()
uk_share = uk_sales / total_sales if total_sales else np.nan

kpi = pd.Series({
    "Rows (clean)": len(df),
    "Invoices": invoices["Invoice"].nunique(),
    "Customers with ID": customers.shape[0],
    "Products (StockCode)": df["StockCode"].nunique(),
    "Countries": df["Country"].nunique(),
    "Date from": df["InvoiceDate"].min().date(),
    "Date to": last_date.date(),
    "Total revenue (GBP)": round(total_sales, 2),
    "Average order value (mean)": round(invoices["revenue"].mean(), 2),
    "Average order value (median)": round(invoices["revenue"].median(), 2),
    "Average units per order": round(invoices["units"].mean(), 1),
    "Average lines per order": round(invoices["lines"].mean(), 1),
    "Repeat customers (>=2 invoices) %": round(100 * customers["is_repeat"].mean(), 2),
    "Revenue from repeat customers %": round(
        100 * customers.loc[customers["is_repeat"], "revenue"].sum() / total_customer_revenue, 2
    ) if total_customer_revenue else np.nan,
    "Top 20pct customers revenue share %": round(100 * top20_share, 2) if pd.notna(top20_share) else np.nan,
    "United Kingdom revenue share %": round(100 * uk_share, 2) if pd.notna(uk_share) else np.nan,
    "Rows missing Customer ID": int(df["Customer ID"].isna().sum()),
}, name="value")
save_table(kpi, "eda_kpis")

# ======================================================================
# 3. MONTHLY TREND
# ======================================================================
monthly = invoices.groupby("month").agg(
    revenue=("revenue", "sum"),
    orders=("Invoice", "nunique"),
    active_customers=("customer", "nunique"),
)
new_per_month = first_month.value_counts().sort_index()
monthly["new_customers"] = new_per_month.reindex(monthly.index).fillna(0).astype(int)
monthly["returning_customers"] = (monthly["active_customers"] - monthly["new_customers"]).clip(lower=0)
monthly["avg_order_value"] = monthly["revenue"] / monthly["orders"].replace(0, np.nan)
monthly["is_partial_month"] = monthly.index == partial_month if partial_month is not None else False
save_table(monthly, "eda_monthly_trend")

labels = [str(p) for p in monthly.index]
bar_colors = [GREY if partial_month is not None and p == partial_month else BLUE for p in monthly.index]
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(labels, monthly["revenue"] / 1000, color=bar_colors)
ax.set_title("Monthly revenue (grey = incomplete last month)")
ax.set_ylabel("Revenue (GBP thousand)")
ax.tick_params(axis="x", rotation=60)
save_fig(fig, "eda_monthly_revenue")

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(labels, monthly["new_customers"], label="New customers", color=ORANGE)
ax.bar(labels, monthly["returning_customers"], bottom=monthly["new_customers"], label="Returning customers", color=BLUE)
ax.set_title("Active customers per month: new vs returning")
ax.set_ylabel("Customers")
ax.tick_params(axis="x", rotation=60)
ax.legend()
save_fig(fig, "eda_new_vs_returning_customers")

# ======================================================================
# 4. COUNTRY
# ======================================================================
country = (
    df.groupby("Country", dropna=False)
    .agg(revenue=("Sales", "sum"), invoices=("Invoice", "nunique"), customers=("Customer ID", "nunique"))
    .sort_values("revenue", ascending=False)
)
country["revenue_share_pct"] = 100 * country["revenue"] / total_sales if total_sales else np.nan
save_table(country, "eda_country_summary")
top_c = country.drop(index="United Kingdom", errors="ignore").head(10)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(top_c.index.astype(str)[::-1], top_c["revenue"].iloc[::-1] / 1000, color=BLUE)
ax.set_title(f"Top 10 countries by revenue excluding UK (UK = {100 * uk_share:.0f}% of total)" if pd.notna(uk_share) else "Top 10 countries by revenue excluding UK")
ax.set_xlabel("Revenue (GBP thousand)")
save_fig(fig, "eda_top_countries_excluding_uk")

# ======================================================================
# 5. PRODUCTS
# ======================================================================
def most_common_description(series):
    modes = series.dropna().astype(str).mode()
    return modes.iloc[0] if not modes.empty else "Unknown product"

names = df.groupby("StockCode")["Description"].agg(most_common_description)
products = (
    df.groupby("StockCode")
    .agg(revenue=("Sales", "sum"), units=("Quantity", "sum"), invoices=("Invoice", "nunique"), customers=("Customer ID", "nunique"))
    .join(names.rename("description"))
    .sort_values("revenue", ascending=False)
)
save_table(products.head(50), "eda_top_products")
top_p = products.head(15)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([str(d)[:38] for d in top_p["description"]][::-1], top_p["revenue"].iloc[::-1] / 1000, color=BLUE)
ax.set_title("Top 15 products by revenue")
ax.set_xlabel("Revenue (GBP thousand)")
save_fig(fig, "eda_top_products_revenue")

# ======================================================================
# 6. CUSTOMER BEHAVIOUR
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
orders_clip = customers["orders"].clip(upper=20)
axes[0].hist(orders_clip, bins=np.arange(0.5, 21.5, 1), color=BLUE, edgecolor="white")
axes[0].set_title("Orders per customer (20+ grouped)")
axes[0].set_xlabel("Number of invoices")
axes[0].set_ylabel("Customers")
axes[1].hist(np.log10(customers["revenue"].clip(lower=1)), bins=40, color=ORANGE, edgecolor="white")
axes[1].set_title("Total spend per customer (log scale)")
axes[1].set_xlabel("log10(GBP)")
share = cust_sorted.cumsum() / total_customer_revenue if total_customer_revenue else cust_sorted * np.nan
axes[2].plot(np.arange(1, len(share) + 1) / len(share) * 100, share.values * 100, color=BLUE)
axes[2].axvline(20, color=GREY, ls="--")
if pd.notna(top20_share):
    axes[2].axhline(100 * top20_share, color=GREY, ls="--")
axes[2].set_title(f"Pareto: top 20% customers = {100 * top20_share:.0f}% revenue" if pd.notna(top20_share) else "Customer revenue concentration")
axes[2].set_xlabel("% of customers (highest spend first)")
axes[2].set_ylabel("% of revenue")
save_fig(fig, "eda_customer_behaviour")

repeat_tbl = customers.groupby("is_repeat").agg(
    customers=("orders", "size"),
    revenue=("revenue", "sum"),
    avg_orders=("orders", "mean"),
    avg_spend=("revenue", "mean"),
)
repeat_tbl.index = repeat_tbl.index.map({False: "One-time", True: "Repeat"})
save_table(repeat_tbl, "eda_repeat_vs_onetime")

# ======================================================================
# 7. TIME-OF-WEEK / TIME-OF-DAY
# ======================================================================
invoices["weekday"] = invoices["date"].dt.day_name()
invoices["hour"] = invoices["date"].dt.hour
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday = (
    invoices.groupby("weekday").agg(orders=("Invoice", "nunique"), revenue=("revenue", "sum"))
    .reindex(weekday_order).fillna(0)
)
hourly = invoices.groupby("hour").agg(orders=("Invoice", "nunique"), revenue=("revenue", "sum"))
save_table(weekday, "eda_weekday")
save_table(hourly, "eda_hourly")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(weekday.index.str[:3], weekday["orders"], color=BLUE)
axes[0].set_title("Orders by weekday")
axes[1].bar(hourly.index, hourly["orders"], color=ORANGE)
axes[1].set_title("Orders by hour of day")
axes[1].set_xlabel("Hour")
save_fig(fig, "eda_weekday_hour")

# ======================================================================
# 8. BASKET SIZE
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(invoices["revenue"].clip(upper=invoices["revenue"].quantile(0.99)), bins=50, color=BLUE, edgecolor="white")
axes[0].set_title("Order value (capped at 99th percentile)")
axes[0].set_xlabel("GBP")
axes[1].hist(invoices["lines"].clip(upper=invoices["lines"].quantile(0.99)), bins=40, color=ORANGE, edgecolor="white")
axes[1].set_title("Product lines per order (capped at 99th percentile)")
axes[1].set_xlabel("Product lines")
save_fig(fig, "eda_basket_size")

# ======================================================================
# 9. COHORT RETENTION
# ======================================================================
act = customer_invoices[["customer", "month"]].drop_duplicates().copy()
act["cohort"] = act["customer"].map(first_month)
# Month difference as integer; avoids Period objects having no `.n` attribute.
act["age"] = [int((m.year - c.year) * 12 + (m.month - c.month)) for m, c in zip(act["month"], act["cohort"])]
cohort_counts = act.pivot_table(index="cohort", columns="age", values="customer", aggfunc="nunique")
if 0 not in cohort_counts.columns:
    raise ValueError("Cohort month 0 is missing; cannot calculate retention.")
retention = cohort_counts.div(cohort_counts[0].replace(0, np.nan), axis=0) * 100
save_table(retention.round(1), "eda_cohort_retention_pct")

view = retention.iloc[:, :13]
fig, ax = plt.subplots(figsize=(12, 8))
if view.size:
    im = ax.imshow(view.values, cmap="Blues", vmin=0, vmax=60, aspect="auto")
    ax.set_xticks(range(view.shape[1]))
    ax.set_yticks(range(view.shape[0]))
    ax.set_yticklabels([str(i) for i in view.index])
    ax.set_xlabel("Months since first purchase")
    ax.set_title("Cohort retention % (first 13 months, including month 0)")
    ax.grid(False)
    for i in range(view.shape[0]):
        for j in range(view.shape[1]):
            value = view.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", color="white" if value > 35 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="% of cohort active")
else:
    ax.text(0.5, 0.5, "No cohort data available", ha="center", va="center")
    ax.set_axis_off()
save_fig(fig, "eda_cohort_retention_heatmap")

# ======================================================================
# 10. WRITTEN SUMMARY (numbers calculated from this dataset)
# ======================================================================
full_monthly = monthly[~monthly["is_partial_month"]]
if full_monthly.empty:
    full_monthly = monthly
best_month = full_monthly["revenue"].idxmax()
top_prod = products.iloc[0]
m1 = retention[1].mean() if 1 in retention.columns else float("nan")

summary_lines = [
    "# EDA Summary - Online Retail II",
    "",
    f"- Period: {kpi['Date from']} to {kpi['Date to']} ({len(df):,} transaction rows, {customers.shape[0]:,} identified customers, {invoices.shape[0]:,} invoices).",
    f"- Total revenue: GBP {kpi['Total revenue (GBP)']:,.2f}; UK revenue share: {100 * uk_share:.1f}%." if pd.notna(uk_share) else f"- Total revenue: GBP {kpi['Total revenue (GBP)']:,.2f}.",
    f"- Average order value: GBP {kpi['Average order value (mean)']:.2f}; median order value: GBP {kpi['Average order value (median)']:.2f}.",
    f"- {kpi['Repeat customers (>=2 invoices) %']:.1f}% of identified customers placed at least two invoices.",
    f"- Repeat customers account for {kpi['Revenue from repeat customers %']:.1f}% of identified-customer revenue." if pd.notna(kpi['Revenue from repeat customers %']) else "- Repeat-customer revenue share could not be calculated.",
    f"- The top 20% of identified customers account for {kpi['Top 20pct customers revenue share %']:.1f}% of identified-customer revenue." if pd.notna(kpi['Top 20pct customers revenue share %']) else "- Customer revenue concentration could not be calculated.",
    f"- Highest-revenue full month: {best_month} (GBP {monthly.loc[best_month, 'revenue']:,.2f}).",
    f"- Highest-revenue product: {top_prod['description']} (StockCode {products.index[0]}; GBP {top_prod['revenue']:,.2f}).",
    f"- Average month-1 cohort retention: {m1:.1f}%." if pd.notna(m1) else "- Month-1 cohort retention could not be calculated.",
    f"- Orders on Saturday: {100 * weekday.loc['Saturday', 'orders'] / weekday['orders'].sum():.1f}% of all invoices." if weekday["orders"].sum() else "- Weekday distribution could not be calculated.",
]
if partial_month is not None:
    summary_lines.append(f"- NOTE: {partial_month} may be incomplete because the latest transaction date is {last_date.date()}; it is excluded from the full-month peak comparison.")
summary_lines += [
    "",
    "## Interpretation notes",
    "- Customer-level metrics exclude transactions/invoices without an identifiable Customer ID; transaction and revenue totals include all usable cleaned rows.",
    "- Cohort retention compares each customer's first observed purchase month with later observed active months. The first cohort can be affected by customers whose earlier history predates this dataset.",
    "- This EDA reads the existing cleaned CSV; cleaning rules are applied by the main project pipeline, not re-created here.",
]
(REPORTS / "eda_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("\n".join(summary_lines))
print(f"\nTables/reports saved to: {REPORTS.resolve()}")
print(f"Figures saved to: {FIGS.resolve()}")
print("EDA completed successfully.")
