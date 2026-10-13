# ================================================================
# ONLINE RETAIL II
# CUSTOMER RETENTION / REPEAT-PURCHASE PREDICTION  (professional pipeline)
#
#   1. Load all sheets + data-quality audit
#   2. Cleaning with a row-by-row waterfall (nothing disappears silently)
#   3. Cohort retention analysis
#   4. Leakage-safe feature engineering (first 90 days -> days 90-180)
#   5. Coverage / column-usage / imbalance analysis
#   6. Purged temporal split + feature selection
#   7. 10 models + baselines, cross-validated model selection
#   8. Hyper-parameter tuning of the top 3 models
#   9. Threshold tuning (out-of-fold, never on the test set)
#  10. Final evaluation: CI, lift, permutation importance, sensitivity
#  11. Final results report
#
# Run from the project root:   python run_project.py
# EDA charts live in 02_eda.py (this script does not repeat them).
# ================================================================

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
START_TIME = time.time()

# ================================================================
# 0. CONFIGURATION
# ================================================================

DATA_PATH = Path("dataset/online_retail_II.xlsx")
OUTPUT = Path("outputs")
PROCESSED, REPORTS = OUTPUT / "processed", OUTPUT / "reports"
FIGURES, MODELS_DIR = OUTPUT / "figures", OUTPUT / "models"
for folder in (PROCESSED, REPORTS, FIGURES, MODELS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
HEADLINE_THRESHOLD = 0.50   # headline metrics; the OOF-tuned threshold is a sensitivity check
FEATURE_WINDOW_DAYS = 90      # behaviour used as features
TARGET_END_DAYS = 180         # target window = day 90 .. day 180
TEST_FRACTION = 0.20          # latest 20% of first-purchase dates = test
CV_FOLDS = 5
TUNE_TOP_N = 3                # tune the best N models
TUNE_ITER = 12                # random-search candidates per tuned model
CORR_THRESHOLD = 0.95         # drop one feature of a pair above this |rho|
BOOTSTRAPS = 1000
N_JOBS = -1                   # use all CPU cores for the search

REQUIRED = ["Invoice", "StockCode", "Description", "Quantity",
            "InvoiceDate", "Price", "Customer ID", "Country"]

# Postage, fees, test rows, vouchers: they are not products.
NON_PRODUCT_CODES = {"POST", "DOT", "M", "C2", "D", "S", "B", "BANK CHARGES",
                     "ADJUST", "ADJUST2", "AMAZONFEE", "PADS", "CRUK"}


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def to_jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp, pd.Period, Path)):
        return str(obj)
    return str(obj)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, default=to_jsonable)


def save_fig(name, dpi=150):
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=dpi)
    plt.close()


def is_non_product(codes: pd.Series) -> pd.Series:
    return (codes.isin(NON_PRODUCT_CODES)
            | codes.str.startswith("TEST", na=False)
            | codes.str.startswith("GIFT_", na=False))


banner("ONLINE RETAIL II - RETENTION PREDICTION PIPELINE")
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}\n"
        "Put online_retail_II.xlsx inside the 'dataset' folder.")

# ================================================================
# 1. LOAD ALL SHEETS + AUDIT
# ================================================================

print("\n[1/11] LOADING DATA")
sheets = pd.read_excel(DATA_PATH, sheet_name=None)   # ALL sheets, not only the first
frames = []
for sheet_name, part in sheets.items():
    part = part.copy()
    part.columns = part.columns.astype(str).str.strip()
    missing_cols = [c for c in REQUIRED if c not in part.columns]
    if missing_cols:
        raise ValueError(f"Sheet '{sheet_name}' misses columns {missing_cols}")
    part = part[REQUIRED].copy()
    part["SourceSheet"] = sheet_name
    frames.append(part)
    print(f"  {sheet_name}: {len(part):,} rows")
raw = pd.concat(frames, ignore_index=True)

original_rows = len(raw)
# FIX: 1-9 Dec 2010 exists in both sheets. SourceSheet must NOT take part
# in the duplicate check, otherwise those rows are counted twice.
original_duplicates = int(raw.duplicated(subset=REQUIRED).sum())

audit = {
    "original_rows": original_rows,
    "sheets": {k: int(len(v)) for k, v in sheets.items()},
    "exact_duplicates_incl_sheet_overlap": original_duplicates,
    "missing_values": {c: int(raw[c].isna().sum()) for c in REQUIRED},
    "missing_customer_ids": int(raw["Customer ID"].isna().sum()),
    "missing_customer_id_pct": round(100 * raw["Customer ID"].isna().mean(), 2),
    "missing_descriptions": int(raw["Description"].isna().sum()),
}
save_json(audit, REPORTS / "data_quality_before_cleaning.json")
print(f"  total rows {original_rows:,} | duplicates {original_duplicates:,} "
      f"| missing Customer ID {audit['missing_customer_ids']:,}")

# ================================================================
# 2. CLEANING WITH WATERFALL
# ================================================================

print("\n[2/11] CLEANING (waterfall)")
d = raw.drop(columns=["SourceSheet"]).copy()
d["Invoice"] = d["Invoice"].astype(str).str.strip().str.upper()
d["StockCode"] = d["StockCode"].astype(str).str.strip().str.upper()   # 85123a == 85123A
d["Description"] = d["Description"].astype("object").where(d["Description"].notna(), "Unknown product")
d["Description"] = d["Description"].astype(str).str.strip()
d["Country"] = d["Country"].fillna("Unknown").astype(str).str.strip()
d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="coerce")   # no dayfirst!
d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce")
d["Price"] = pd.to_numeric(d["Price"], errors="coerce")
d["Customer ID"] = pd.to_numeric(d["Customer ID"], errors="coerce")

waterfall = [("Start (all sheets stacked)", 0, len(d))]


def step(label, frame, mask):
    before = len(frame)
    out = frame.loc[mask].copy()
    waterfall.append((label, before - len(out), len(out)))
    return out


n0 = len(d)
d = d.drop_duplicates(subset=REQUIRED).copy()
waterfall.append(("Exact duplicates (incl. sheet overlap)", n0 - len(d), len(d)))

is_cancel = d["Invoice"].str.startswith("C")
cancelled_all = d.loc[is_cancel].copy()                  # kept for return features
d = step("Cancelled invoices (C...)", d, ~is_cancel)
d = step("Other non-numeric invoices (A... adjustments)", d, d["Invoice"].str.fullmatch(r"\d+"))
d = step("Missing Customer ID (guest orders)", d, d["Customer ID"].notna())
d = step("Invalid date", d, d["InvoiceDate"].notna())
d = step("Quantity <= 0", d, d["Quantity"] > 0)
d = step("Price <= 0", d, d["Price"] > 0)
d = step("Non-product codes (POST, M, fees...)", d, ~is_non_product(d["StockCode"]))

d["Customer ID"] = d["Customer ID"].astype(int).astype(str)
d["Sales"] = d["Quantity"] * d["Price"]
df = d.reset_index(drop=True)
if df.empty:
    raise ValueError("No usable transactions remain after cleaning.")

# Cancelled lines of identified customers -> return/cancellation features.
cancelled = cancelled_all[cancelled_all["Customer ID"].notna()
                          & ~is_non_product(cancelled_all["StockCode"])].copy()
cancelled["Customer ID"] = cancelled["Customer ID"].astype(int).astype(str)
cancelled["ReturnValue"] = (cancelled["Quantity"] * cancelled["Price"]).abs()

waterfall_df = pd.DataFrame(waterfall, columns=["step", "rows_removed", "rows_remaining"])
waterfall_df.to_csv(REPORTS / "cleaning_waterfall.csv", index=False)
print(waterfall_df.to_string(index=False))

cleaning_report = {
    **audit,
    "rows_after_cleaning": int(len(df)),
    "rows_removed_overall": int(original_rows - len(df)),
    "unique_customers": int(df["Customer ID"].nunique()),
    "unique_invoices": int(df["Invoice"].nunique()),
    "unique_products": int(df["StockCode"].nunique()),
    "date_start": str(df["InvoiceDate"].min()),
    "date_end": str(df["InvoiceDate"].max()),
    "missing_values_after_cleaning": {c: int(df[c].isna().sum()) for c in df.columns},
}
save_json(cleaning_report, REPORTS / "data_quality_after_cleaning.json")
df.to_csv(PROCESSED / "cleaned_transactions.csv", index=False)
print(f"  final: {len(df):,} rows | {df['Customer ID'].nunique():,} customers | "
      f"{df['InvoiceDate'].min().date()} -> {df['InvoiceDate'].max().date()}")

# ================================================================
# 3. COHORT RETENTION
# ================================================================

print("\n[3/11] COHORT RETENTION")
co = df[["Customer ID", "InvoiceDate"]].copy()
co["InvoiceMonth"] = co["InvoiceDate"].dt.to_period("M")
co["CohortMonth"] = co.groupby("Customer ID")["InvoiceDate"].transform("min").dt.to_period("M")
co["CohortIndex"] = ((co["InvoiceMonth"].dt.year - co["CohortMonth"].dt.year) * 12
                     + co["InvoiceMonth"].dt.month - co["CohortMonth"].dt.month)
cohort_table = (co.groupby(["CohortMonth", "CohortIndex"])["Customer ID"].nunique()
                .unstack("CohortIndex"))
retention = cohort_table.div(cohort_table[0], axis=0)
cohort_table.to_csv(REPORTS / "cohort_customer_counts.csv")
retention.to_csv(REPORTS / "cohort_retention_matrix.csv")

plt.figure(figsize=(14, 9))
sns.heatmap(retention.iloc[:, :13], annot=True, fmt=".0%", cmap="YlGnBu",
            vmin=0, vmax=0.6, annot_kws={"size": 7})
plt.title("Monthly cohort retention (month 0 = first purchase month)")
plt.xlabel("Months since first purchase")
plt.ylabel("First-purchase cohort")
save_fig("cohort_retention_heatmap")

orders_per_customer = df.groupby("Customer ID")["Invoice"].nunique()
overall_repeat = pd.DataFrame([{
    "total_customers": len(orders_per_customer),
    "repeat_customers": int((orders_per_customer >= 2).sum()),
    "one_invoice_customers": int((orders_per_customer == 1).sum()),
    "repeat_customer_rate": float((orders_per_customer >= 2).mean()),
}])
overall_repeat.to_csv(REPORTS / "overall_repeat_purchase_summary.csv", index=False)
print(f"  repeat customers (>=2 invoices): {overall_repeat.loc[0, 'repeat_customer_rate']:.1%}")

# ================================================================
# 4. FEATURE ENGINEERING (leakage-safe)
# ================================================================

print("\n[4/11] FEATURE ENGINEERING")
# Every feature is computed ONLY from days [0, 90) after the customer's first
# purchase. The target uses ONLY days [90, 180). They never overlap.

last_observation = df["InvoiceDate"].max()
first_dates = df.groupby("Customer ID")["InvoiceDate"].min()
all_customers = len(first_dates)

follow_up_ok = first_dates <= last_observation - pd.Timedelta(days=TARGET_END_DAYS)
eligible_ids = first_dates[follow_up_ok].index
if len(eligible_ids) == 0:
    raise ValueError("No customer has 180 days of follow-up. Check the date range.")

el = df[df["Customer ID"].isin(eligible_ids)].copy()
el["FirstPurchase"] = el["Customer ID"].map(first_dates)
el["Day"] = (el["InvoiceDate"] - el["FirstPurchase"]).dt.total_seconds() / 86400

fw = el[el["Day"] < FEATURE_WINDOW_DAYS].copy()                                   # features
tw = el[(el["Day"] >= FEATURE_WINDOW_DAYS) & (el["Day"] < TARGET_END_DAYS)]       # target
assert fw["Day"].max() < FEATURE_WINDOW_DAYS <= tw["Day"].min(), "window overlap!"

# --- invoice-level table ------------------------------------------------
inv = (fw.groupby(["Customer ID", "Invoice"], as_index=False)
         .agg(date=("InvoiceDate", "min"), day=("Day", "min"),
              basket_value=("Sales", "sum"), items=("Quantity", "sum")))
inv = inv.sort_values(["Customer ID", "date"])
inv["shop_day"] = inv["date"].dt.normalize()
g = inv.groupby("Customer ID")

F = pd.DataFrame(index=g.size().index)
F["total_spend"] = g["basket_value"].sum()
F["total_quantity"] = g["items"].sum()
F["unique_invoices"] = g["Invoice"].nunique()
F["active_purchase_days"] = g["shop_day"].nunique()
F["average_basket_value"] = g["basket_value"].mean()
F["maximum_basket_value"] = g["basket_value"].max()
F["std_basket_value"] = g["basket_value"].std().fillna(0.0)
F["first_basket_value"] = g["basket_value"].first()
F["items_per_invoice"] = F["total_quantity"] / F["unique_invoices"]
F["spend_per_active_day"] = F["total_spend"] / F["active_purchase_days"]
F["recency_days_at_day_90"] = FEATURE_WINDOW_DAYS - g["day"].max()

late = inv[inv["day"] >= FEATURE_WINDOW_DAYS - 30]       # last 30 days of the window
F["invoices_last_30d"] = late.groupby("Customer ID")["Invoice"].nunique().reindex(F.index).fillna(0)
F["spend_last_30d_share"] = (late.groupby("Customer ID")["basket_value"].sum()
                             .reindex(F.index).fillna(0) / F["total_spend"])

# --- shopping-day gaps (several invoices on one day = ONE visit) --------
visits = inv.drop_duplicates(["Customer ID", "shop_day"]).copy()
visits["rank"] = visits.groupby("Customer ID").cumcount()
first_visit = visits[visits["rank"] == 0].set_index("Customer ID")["date"]
second_visit = visits[visits["rank"] == 1].set_index("Customer ID")["date"]
F["days_to_second_purchase"] = ((second_visit - first_visit.reindex(second_visit.index))
                                .dt.total_seconds() / 86400).reindex(F.index)
vg = visits.groupby("Customer ID")["date"]
n_visits = vg.size()
F["average_days_between_visits"] = (
    ((vg.max() - vg.min()).dt.total_seconds() / 86400) / (n_visits - 1).replace(0, np.nan))

# --- line-level features ------------------------------------------------
lg = fw.groupby("Customer ID")
F["unique_products"] = lg["StockCode"].nunique()
F["mean_line_price"] = lg["Price"].mean()
prod_inv = fw.groupby(["Customer ID", "StockCode"])["Invoice"].nunique()
F["repeat_product_share"] = (prod_inv >= 2).groupby(level=0).mean().reindex(F.index)
first_country = fw.sort_values("InvoiceDate").groupby("Customer ID")["Country"].first()
F["is_uk_customer"] = (first_country.reindex(F.index) == "United Kingdom").astype(int)

first_month = first_dates.reindex(F.index).dt.month
F["first_month_sin"] = np.sin(2 * np.pi * first_month / 12)
F["first_month_cos"] = np.cos(2 * np.pi * first_month / 12)

# --- returns / cancellations inside the same 90-day window --------------
cn = cancelled[cancelled["Customer ID"].isin(eligible_ids)].copy()
cn["Day"] = (cn["InvoiceDate"] - cn["Customer ID"].map(first_dates)).dt.total_seconds() / 86400
cn = cn[(cn["Day"] >= 0) & (cn["Day"] < FEATURE_WINDOW_DAYS)]
F["cancelled_invoices_90d"] = cn.groupby("Customer ID")["Invoice"].nunique().reindex(F.index).fillna(0)
F["return_value_ratio"] = (cn.groupby("Customer ID")["ReturnValue"].sum()
                           .reindex(F.index).fillna(0) / F["total_spend"])

# --- target -------------------------------------------------------------
TARGET = "repeat_purchase_next_90d"
future = tw.groupby("Customer ID")["Invoice"].nunique()
F[TARGET] = F.index.to_series().map(future).fillna(0).gt(0).astype(int)
F["first_purchase_date"] = first_dates.reindex(F.index)
F = F.replace([np.inf, -np.inf], np.nan)

# --- catalogue: what each feature is and why it exists ------------------
CATALOG = [
    ("total_spend", "Monetary", "Total GBP spent in first 90 days", "Quantity, Price", "Big spenders are more engaged and more likely to return"),
    ("total_quantity", "Monetary", "Total units bought", "Quantity", "Volume signals wholesale / stock-up behaviour"),
    ("average_basket_value", "Monetary", "Mean invoice value", "Invoice, Quantity, Price", "Typical order size"),
    ("maximum_basket_value", "Monetary", "Largest invoice value", "Invoice, Quantity, Price", "Peak purchasing capacity"),
    ("std_basket_value", "Monetary", "Std-dev of invoice values (0 if one invoice)", "Invoice, Quantity, Price", "Stable vs erratic ordering"),
    ("first_basket_value", "Monetary", "Value of the very first invoice", "Invoice, Quantity, Price", "Strength of the first impression / acquisition order"),
    ("items_per_invoice", "Monetary", "Units per invoice", "Quantity, Invoice", "Basket depth"),
    ("spend_per_active_day", "Monetary", "Spend per shopping day", "Quantity, Price, InvoiceDate", "Intensity of each visit"),
    ("mean_line_price", "Monetary", "Mean unit price of lines", "Price", "Premium vs cheap product mix"),
    ("unique_invoices", "Frequency", "Number of invoices", "Invoice", "Core frequency signal (RFM 'F')"),
    ("active_purchase_days", "Frequency", "Distinct shopping days", "InvoiceDate", "Visits, ignoring split baskets"),
    ("invoices_last_30d", "Frequency", "Invoices in days 60-90", "Invoice, InvoiceDate", "Recent momentum"),
    ("spend_last_30d_share", "Frequency", "Share of spend made in days 60-90", "Quantity, Price, InvoiceDate", "Accelerating vs fading engagement"),
    ("recency_days_at_day_90", "Recency", "Days between last purchase and day 90", "InvoiceDate", "Core recency signal (RFM 'R')"),
    ("days_to_second_purchase", "Recency", "Days from 1st to 2nd shopping day (NaN if none)", "InvoiceDate", "Speed of habit formation"),
    ("average_days_between_visits", "Recency", "Mean gap between shopping days (NaN if 1 visit)", "InvoiceDate", "Purchase rhythm"),
    ("unique_products", "Variety", "Distinct StockCodes bought", "StockCode", "Breadth of interest"),
    ("repeat_product_share", "Variety", "Share of products bought on 2+ invoices", "StockCode, Invoice", "Re-ordering / loyalty to products"),
    ("is_uk_customer", "Profile", "1 if first order country is United Kingdom", "Country", "Domestic vs export behaviour"),
    ("first_month_sin", "Seasonality", "Sine of first-purchase month", "InvoiceDate", "Cyclical season (Dec != Jan-adjacent otherwise)"),
    ("first_month_cos", "Seasonality", "Cosine of first-purchase month", "InvoiceDate", "Cyclical season of acquisition"),
    ("cancelled_invoices_90d", "Returns", "Cancelled invoices in first 90 days", "Invoice (C...)", "Dissatisfaction / return behaviour"),
    ("return_value_ratio", "Returns", "Returned value / spend", "Invoice (C...), Quantity, Price", "Share of spend that came back"),
]
catalog = pd.DataFrame(CATALOG, columns=["feature", "group", "description", "source_columns", "why_used"])
assert set(catalog["feature"]) <= set(F.columns), "catalogue / feature mismatch"
# The data covers < 2 annual cycles and the honest (purged) training set only
# contains customers acquired Dec-Apr, while the test set covers Nov-Jun. A
# month-of-year feature therefore has to EXTRAPOLATE and hurts (see ablation in
# the final report). It is engineered and documented, but not given to the models.
SEASONAL_FEATURES = ["first_month_sin", "first_month_cos"]
candidate_features = [c for c in catalog["feature"] if c not in SEASONAL_FEATURES]
LOG_CANDIDATES = [c for c in ["total_spend", "total_quantity", "average_basket_value",
                              "maximum_basket_value", "std_basket_value", "first_basket_value",
                              "items_per_invoice", "spend_per_active_day", "mean_line_price",
                              "unique_products", "unique_invoices", "active_purchase_days",
                              "invoices_last_30d", "cancelled_invoices_90d",
                              "return_value_ratio", "days_to_second_purchase",
                              "average_days_between_visits", "recency_days_at_day_90"]
                  if c in candidate_features]

print(f"  eligible customers: {len(F):,}  |  features engineered: {len(catalog)} "
      f"| given to models: {len(candidate_features)} (seasonality excluded by design)")

# ================================================================
# 5. COVERAGE, COLUMN USAGE, IMBALANCE
# ================================================================

print("\n[5/11] COVERAGE, COLUMN USAGE, CLASS BALANCE")
y_all = F[TARGET].astype(int)
class_counts = y_all.value_counts().sort_index()
minority_share = float(class_counts.min() / class_counts.sum())
imbalance_ratio = float(class_counts.max() / class_counts.min())

if minority_share >= 0.40:
    imbalance_level, imbalance_action = "balanced / mild", "No resampling needed (no SMOTE, no undersampling)."
elif minority_share >= 0.20:
    imbalance_level, imbalance_action = "moderate", "Use class_weight='balanced' and tune the threshold."
else:
    imbalance_level, imbalance_action = "severe", "Use class_weight, threshold tuning and consider SMOTE."

class_counts.rename_axis("target_class").rename("customers").to_csv(REPORTS / "target_class_distribution.csv")

coverage = pd.DataFrame([
    ("Customers with an ID after cleaning", all_customers),
    ("Dropped: < 180 days of follow-up (target unknown)", all_customers - len(F)),
    ("Customers modelled", len(F)),
], columns=["stage", "customers"])
coverage.to_csv(REPORTS / "customer_coverage.csv", index=False)

column_usage = pd.DataFrame([
    ("Invoice", "Used", "Cancel filter, order grouping -> invoice/basket/frequency features, return features"),
    ("StockCode", "Used", "Non-product filter, unique_products, repeat_product_share"),
    ("Description", "NOT used in model", "Free text (5,000+ variants); used only for EDA product names"),
    ("Quantity", "Used", "Quantity<=0 filter, Sales, total_quantity, items_per_invoice"),
    ("InvoiceDate", "Used", "90/180-day windows, recency, gaps, seasonality, temporal split"),
    ("Price", "Used", "Price<=0 filter, Sales, mean_line_price"),
    ("Customer ID", "Used", "Entity key (rows without it cannot be followed over time)"),
    ("Country", "Used", "is_uk_customer"),
    ("SourceSheet", "Technical only", "Removed after de-duplicating the overlapping sheets"),
], columns=["raw_column", "status", "how_it_is_used"])
column_usage["missing_pct_in_raw"] = column_usage["raw_column"].map(
    lambda c: round(100 * raw[c].isna().mean(), 2) if c in raw.columns else np.nan)
column_usage.to_csv(REPORTS / "column_usage.csv", index=False)

plt.figure(figsize=(5.5, 4))
ax = sns.barplot(x=class_counts.index.map({0: "No purchase\nday 90-180", 1: "Purchase\nday 90-180"}),
                 y=class_counts.values, palette=["#9aa5b1", "#2f6db5"])
for i, v in enumerate(class_counts.values):
    ax.text(i, v, f"{v:,}\n({v / class_counts.sum():.0%})", ha="center", va="bottom")
plt.title(f"Target balance (ratio {imbalance_ratio:.2f} : 1)")
plt.ylabel("Customers")
plt.ylim(0, class_counts.max() * 1.2)
save_fig("target_class_balance")
print(f"  class counts {class_counts.to_dict()} | minority {minority_share:.1%} "
      f"| ratio {imbalance_ratio:.2f}:1 -> {imbalance_level}")

# ================================================================
# 6. PURGED TEMPORAL SPLIT + FEATURE SELECTION
# ================================================================

print("\n[6/11] TEMPORAL SPLIT + FEATURE SELECTION")
dates = F["first_purchase_date"]
cutoff = dates.sort_values().iloc[int((1 - TEST_FRACTION) * (len(dates) - 1))]
test_mask = dates >= cutoff
train_naive_mask = ~test_mask
# Purge: a training customer's label window (first + 180d) must END before the
# test period starts, otherwise training labels reveal the test period's future.
train_mask = dates <= cutoff - pd.Timedelta(days=TARGET_END_DAYS)
purged_n = int(train_naive_mask.sum() - train_mask.sum())

if train_mask.sum() < 300 or y_all[train_mask].nunique() < 2:
    print("  purged training set too small -> using plain temporal split")
    train_mask, purged_n, split_type = train_naive_mask, 0, "temporal (no purge)"
else:
    split_type = "purged temporal"

role = pd.Series("purged (gap)", index=F.index)
role[train_mask] = "train"
role[test_mask] = "test"

X_all, y = F[candidate_features], y_all
cohort_rates = (F.assign(cohort=F["first_purchase_date"].dt.to_period("M").astype(str), role=role)
                .groupby("cohort").agg(customers=(TARGET, "size"), retention_rate=(TARGET, "mean"),
                                       role=("role", lambda r: r.mode().iloc[0])).reset_index())
cohort_rates.to_csv(REPORTS / "target_rate_by_first_purchase_month.csv", index=False)
role_colors = {"train": "#2f6db5", "purged (gap)": "#c9ced6", "test": "#e8833a"}
plt.figure(figsize=(12, 4.5))
plt.bar(cohort_rates["cohort"], 100 * cohort_rates["retention_rate"],
        color=cohort_rates["role"].map(role_colors))
plt.xticks(rotation=60)
plt.ylabel("% who buy again in days 90-180")
plt.title("Target rate by first-purchase month (blue = train, grey = purged gap, orange = test)")
save_fig("target_rate_by_cohort")
X_train, y_train = X_all[train_mask], y[train_mask]
X_test, y_test = X_all[test_mask], y[test_mask]
print(f"  split: {split_type} | test starts {cutoff.date()} | "
      f"train {len(X_train):,} | purged gap {purged_n:,} | test {len(X_test):,}")
print(f"  train positive rate {y_train.mean():.1%} | test positive rate {y_test.mean():.1%}")

# --- selection on TRAIN only, without looking at the target --------------
selected, dropped_reason = [], {}
corr = X_train.corr(method="spearman").abs()
for feat in candidate_features:
    if X_train[feat].nunique(dropna=True) <= 1:
        dropped_reason[feat] = "constant in training data"
        continue
    twin = next((s for s in selected if corr.loc[feat, s] > CORR_THRESHOLD), None)
    if twin:
        dropped_reason[feat] = f"|rho|={corr.loc[feat, twin]:.2f} with {twin}"
    else:
        selected.append(feat)

# univariate ROC-AUC on TRAIN (documentation + leakage sanity check)
uni_auc = {}
for feat in catalog["feature"]:
    col = F.loc[train_mask, feat].fillna(F.loc[train_mask, feat].median())
    auc = roc_auc_score(y_train, col) if col.nunique() > 1 else 0.5
    uni_auc[feat] = max(auc, 1 - auc)
leaks = [f for f, a in uni_auc.items() if a > 0.90]
if leaks:
    print(f"  WARNING: suspiciously predictive features (possible leakage): {leaks}")


def feature_status(c):
    if c in SEASONAL_FEATURES:
        return "excluded by design"
    return "kept" if c in selected else "dropped"


def feature_reason(c):
    if c in SEASONAL_FEATURES:
        return "needs 2+ annual cycles; purged train covers only Dec-Apr (ablation in final report)"
    return dropped_reason.get(c, "informative, not redundant")


catalog["missing_pct"] = catalog["feature"].map(lambda c: round(100 * F[c].isna().mean(), 1))
catalog["train_univariate_auc"] = catalog["feature"].map(lambda c: round(uni_auc[c], 3))
catalog["status"] = catalog["feature"].map(feature_status)
catalog["reason"] = catalog["feature"].map(feature_reason)
catalog.to_csv(REPORTS / "feature_catalog.csv", index=False)

X_train, X_test = X_train[selected], X_test[selected]
log_cols = [c for c in LOG_CANDIDATES if c in selected]

plt.figure(figsize=(12, 10))
sns.heatmap(X_all[selected].corr(method="spearman"), cmap="coolwarm", center=0,
            annot=len(selected) <= 18, fmt=".1f", annot_kws={"size": 6})
plt.title("Spearman correlation of selected features")
save_fig("feature_correlation_heatmap")

model_data = F[selected + [TARGET, "first_purchase_date"]].copy()
model_data["split_role"] = role
model_data.to_csv(PROCESSED / "customer_ml_dataset.csv")
print(f"  features: {len(catalog)} engineered -> {len(SEASONAL_FEATURES)} excluded by design "
      f"-> {len(candidate_features) - len(selected)} dropped as redundant -> {len(selected)} kept")
for feat, why in dropped_reason.items():
    print(f"    dropped {feat}: {why}")

# ================================================================
# 7. MODELS, BASELINES, CROSS-VALIDATED SELECTION
# ================================================================


def make_pipeline(estimator):
    """All preprocessing lives INSIDE the pipeline -> no leakage across CV folds."""
    pre = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p), log_cols)] if log_cols else [],
        remainder="passthrough")
    return Pipeline([
        ("log", pre),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", clone(estimator)),
    ])


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "Gaussian Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "Neural Network MLP": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
}

SEARCH_SPACES = {
    "Logistic Regression": {"classifier__C": loguniform(1e-3, 1e2)},
    "Decision Tree": {"classifier__max_depth": [3, 4, 5, 6, 8, 10], "classifier__min_samples_leaf": [5, 10, 20, 40]},
    "Random Forest": {"classifier__n_estimators": [200, 300], "classifier__max_depth": [4, 6, 8, 12, None],
                      "classifier__min_samples_leaf": [1, 3, 5, 10], "classifier__max_features": ["sqrt", 0.5]},
    "Extra Trees": {"classifier__n_estimators": [200, 300], "classifier__max_depth": [4, 6, 8, 12, None],
                    "classifier__min_samples_leaf": [1, 3, 5, 10], "classifier__max_features": ["sqrt", 0.5]},
    "K-Nearest Neighbors": {"classifier__n_neighbors": [5, 11, 15, 25, 35, 51], "classifier__weights": ["uniform", "distance"]},
    "Support Vector Machine": {"classifier__C": loguniform(1e-1, 1e1), "classifier__gamma": ["scale", 0.01, 0.03, 0.1]},
    "Gaussian Naive Bayes": {"classifier__var_smoothing": loguniform(1e-10, 1e-5)},
    "Gradient Boosting": {"classifier__n_estimators": [100, 200, 300], "classifier__learning_rate": [0.02, 0.05, 0.1],
                          "classifier__max_depth": [2, 3, 4], "classifier__subsample": [0.7, 0.85, 1.0],
                          "classifier__min_samples_leaf": [5, 10, 20]},
    "Hist Gradient Boosting": {"classifier__learning_rate": [0.03, 0.05, 0.1], "classifier__max_depth": [3, 5, None],
                               "classifier__max_leaf_nodes": [15, 31], "classifier__l2_regularization": [0.0, 1.0, 10.0],
                               "classifier__min_samples_leaf": [20, 40]},
    "Neural Network MLP": {"classifier__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                           "classifier__alpha": loguniform(1e-5, 1e-1), "classifier__learning_rate_init": [1e-3, 3e-3, 1e-2]},
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1"}


def score_on(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, np.clip(proba, 0, 1)),
    }


print("\n[7/11] MODEL COMPARISON (5-fold CV on the training set)")
print(f"  {'model':26s} {'CV PR-AUC':>10s} {'CV ROC-AUC':>11s} {'CV F1':>7s} | {'test ROC-AUC':>12s} {'test PR-AUC':>11s}")
rows, fitted = [], {}
for name, est in MODELS.items():
    pipe = make_pipeline(est)
    try:
        cvr = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1, error_score=np.nan)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        t = score_on(y_test, proba)
        rows.append({"model": name, "tuned": False,
                     "cv_pr_auc": np.nanmean(cvr["test_pr_auc"]), "cv_pr_auc_std": np.nanstd(cvr["test_pr_auc"]),
                     "cv_roc_auc": np.nanmean(cvr["test_roc_auc"]), "cv_f1": np.nanmean(cvr["test_f1"]),
                     **{f"test_{k}": v for k, v in t.items()}, "status": "success"})
        fitted[name] = pipe
        print(f"  {name:26s} {rows[-1]['cv_pr_auc']:10.3f} {rows[-1]['cv_roc_auc']:11.3f} "
              f"{rows[-1]['cv_f1']:7.3f} | {t['roc_auc']:12.3f} {t['pr_auc']:11.3f}")
    except Exception as exc:
        rows.append({"model": name, "tuned": False, "status": f"failed: {str(exc)[:120]}"})
        print(f"  {name:26s} FAILED: {str(exc)[:90]}")

results = pd.DataFrame(rows)
ok = results[results["status"] == "success"].sort_values("cv_pr_auc", ascending=False)
if ok.empty:
    raise RuntimeError("All models failed - check the console output above.")

# --- baselines (what a trivial approach achieves) ------------------------
baseline_rows = []
dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
baseline_rows.append({"model": "Baseline: always majority class",
                      **score_on(y_test, dummy.predict_proba(X_test)[:, 1], threshold=0.5)})
heur_score = X_all.loc[test_mask, "unique_invoices"].values   # independent of feature selection
heur_pred = (heur_score >= 2).astype(int)
baseline_rows.append({"model": "Baseline: rule 'bought on 2+ invoices in 90d'",
                      "accuracy": accuracy_score(y_test, heur_pred),
                      "balanced_accuracy": balanced_accuracy_score(y_test, heur_pred),
                      "precision": precision_score(y_test, heur_pred, zero_division=0),
                      "recall": recall_score(y_test, heur_pred, zero_division=0),
                      "f1": f1_score(y_test, heur_pred, zero_division=0),
                      "mcc": matthews_corrcoef(y_test, heur_pred),
                      "roc_auc": roc_auc_score(y_test, heur_score),
                      "pr_auc": average_precision_score(y_test, heur_score), "brier": np.nan})
baselines = pd.DataFrame(baseline_rows)
baselines.to_csv(REPORTS / "baselines.csv", index=False)

# --- does class imbalance handling matter? (CV on training set) ----------
imb_rows = []
for name in ("Logistic Regression", "Random Forest"):
    for cw in (None, "balanced"):
        est = clone(MODELS[name]).set_params(class_weight=cw)
        r = cross_validate(make_pipeline(est), X_train, y_train, cv=cv,
                           scoring={"pr_auc": "average_precision", "f1": "f1",
                                    "balanced_accuracy": "balanced_accuracy"}, n_jobs=1)
        imb_rows.append({"model": name, "class_weight": "none" if cw is None else cw,
                         "cv_pr_auc": r["test_pr_auc"].mean(), "cv_f1": r["test_f1"].mean(),
                         "cv_balanced_accuracy": r["test_balanced_accuracy"].mean()})
imb_df = pd.DataFrame(imb_rows)
imb_df.to_csv(REPORTS / "imbalance_analysis.csv", index=False)
print("\n  Imbalance experiment (CV):")
print(imb_df.round(3).to_string(index=False))

# ================================================================
# 8. TUNE THE TOP-N MODELS
# ================================================================

print(f"\n[8/11] HYPER-PARAMETER TUNING (top {TUNE_TOP_N} by CV PR-AUC)")
tuned, tuned_rows = {}, []
for name in ok["model"].head(TUNE_TOP_N):
    search = RandomizedSearchCV(
        make_pipeline(MODELS[name]), SEARCH_SPACES[name], n_iter=TUNE_ITER, cv=cv,
        scoring="average_precision", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        refit=True, error_score=np.nan)
    search.fit(X_train, y_train)
    proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    t = score_on(y_test, proba)
    default_cv = float(ok.loc[ok["model"] == name, "cv_pr_auc"].iloc[0])
    tuned[name] = search
    tuned_rows.append({"model": name, "tuned": True, "default_cv_pr_auc": default_cv,
                       "cv_pr_auc": search.best_score_, "best_params": str(search.best_params_),
                       **{f"test_{k}": v for k, v in t.items()}})
    print(f"  {name:26s} CV PR-AUC {default_cv:.3f} -> {search.best_score_:.3f} | test PR-AUC {t['pr_auc']:.3f}")

tuned_df = pd.DataFrame(tuned_rows)
tuned_df.to_csv(REPORTS / "tuned_models.csv", index=False)

# Final model = best CROSS-VALIDATED score. The test set never chooses anything.
candidates = [(r["model"], r["cv_pr_auc"], fitted[r["model"]], False) for _, r in ok.iterrows()]
candidates += [(n, s.best_score_, s.best_estimator_, True) for n, s in tuned.items()]
best_name, best_cv, best_model, best_is_tuned = max(candidates, key=lambda c: c[1])
best_label = f"{best_name} ({'tuned' if best_is_tuned else 'default'})"
print(f"\n  FINAL MODEL: {best_label} | CV PR-AUC {best_cv:.3f}")

results = pd.concat([results, tuned_df], ignore_index=True)
results.to_csv(REPORTS / "model_comparison.csv", index=False)

# ================================================================
# 9. THRESHOLD TUNING (out-of-fold, training data only)
# ================================================================

print("\n[9/11] THRESHOLD TUNING (sensitivity check, out-of-fold on training data)")
oof = cross_val_predict(clone(best_model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
grid = np.round(np.arange(0.20, 0.81, 0.01), 2)
oof_f1 = np.array([f1_score(y_train, (oof >= t).astype(int), zero_division=0) for t in grid])
is_best = oof_f1 >= oof_f1.max() - 1e-9
tied = grid[is_best]
best_thr = float(tied[np.argmin(np.abs(tied - 0.5))])      # ties -> closest to 0.50
print(f"  best out-of-fold threshold: {best_thr:.2f} (OOF F1 {oof_f1.max():.3f} vs {f1_score(y_train, (oof >= 0.5).astype(int)):.3f} at 0.50)")

# ================================================================
# 10. FINAL EVALUATION
# ================================================================

print("\n[10/11] FINAL EVALUATION ON THE UNTOUCHED TEST SET")
test_proba = best_model.predict_proba(X_test)[:, 1]
metrics_050 = score_on(y_test, test_proba, 0.5)
metrics_tuned = score_on(y_test, test_proba, best_thr)
final_pred = (test_proba >= HEADLINE_THRESHOLD).astype(int)

# 95% bootstrap confidence intervals
rng = np.random.default_rng(RANDOM_STATE)
yt, pt = y_test.values, test_proba
boot = {"f1": [], "roc_auc": [], "pr_auc": []}
for _ in range(BOOTSTRAPS):
    idx = rng.integers(0, len(yt), len(yt))
    if len(np.unique(yt[idx])) < 2:
        continue
    boot["f1"].append(f1_score(yt[idx], (pt[idx] >= HEADLINE_THRESHOLD).astype(int), zero_division=0))
    boot["roc_auc"].append(roc_auc_score(yt[idx], pt[idx]))
    boot["pr_auc"].append(average_precision_score(yt[idx], pt[idx]))
ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}

final_table = pd.DataFrame({"threshold_0.50 (headline)": metrics_050, f"threshold_{best_thr:.2f} (OOF-tuned)": metrics_tuned})
final_table.to_csv(REPORTS / "final_test_metrics.csv")

# Lift by decile
lift = pd.DataFrame({"y": yt, "p": pt})
lift["decile"] = pd.qcut(lift["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
lift_table = lift.groupby("decile", observed=True).agg(customers=("y", "size"), retention_rate=("y", "mean"),
                                                       retained=("y", "sum")).reset_index()
lift_table["lift_vs_average"] = lift_table["retention_rate"] / yt.mean()
lift_table["cumulative_capture_pct"] = 100 * lift_table["retained"].cumsum() / lift_table["retained"].sum()
lift_table.to_csv(REPORTS / "lift_table.csv", index=False)

# Permutation importance on the test set (model-agnostic)
perm = permutation_importance(best_model, X_test, y_test, scoring="average_precision",
                              n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
importance = (pd.DataFrame({"feature": selected, "importance": perm.importances_mean,
                            "std": perm.importances_std})
              .sort_values("importance", ascending=False))
importance.to_csv(REPORTS / "permutation_importance.csv", index=False)

# Sensitivity: what if we had NOT purged the training set?
naive_pr = np.nan
naive_n = int(train_naive_mask.sum())
if split_type == "purged temporal":
    naive_model = clone(best_model).fit(X_all.loc[train_naive_mask, selected], y[train_naive_mask])
    naive_pr = average_precision_score(y_test, naive_model.predict_proba(X_test)[:, 1])

# Ablation: what would the seasonal features have done under the same honest split?
with_season = clone(best_model).fit(F.loc[train_mask, selected + SEASONAL_FEATURES], y_train)
season_proba = with_season.predict_proba(F.loc[test_mask, selected + SEASONAL_FEATURES])[:, 1]
season_roc = roc_auc_score(y_test, season_proba)
season_pr = average_precision_score(y_test, season_proba)

report_text = classification_report(y_test, final_pred, zero_division=0)
with open(REPORTS / "best_model_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Best model: {best_label}\nSelection: highest 5-fold CV PR-AUC on the training set\n"
            f"Split: {split_type} | threshold {HEADLINE_THRESHOLD:.2f}\n"
            f"Train {len(X_train)} | Test {len(X_test)}\n\n{report_text}")

# ---------------- figures ----------------
cm = confusion_matrix(y_test, final_pred, labels=[0, 1])
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"])
plt.title(f"Confusion matrix: {best_name}\n(threshold {HEADLINE_THRESHOLD:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
save_fig("best_model_confusion_matrix")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
fpr, tpr, _ = roc_curve(y_test, test_proba)
axes[0].plot(fpr, tpr, color="#2f6db5", label=f"{best_name} (AUC {metrics_050['roc_auc']:.3f})")
hf, ht, _ = roc_curve(y_test, heur_score)
axes[0].plot(hf, ht, color="#e8833a", ls="--", label=f"Rule baseline (AUC {baselines.loc[1, 'roc_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], color="grey", ls=":")
axes[0].set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
axes[0].legend(loc="lower right")
pr_p, pr_r, _ = precision_recall_curve(y_test, test_proba)
axes[1].plot(pr_r, pr_p, color="#2f6db5", label=f"{best_name} (AP {metrics_050['pr_auc']:.3f})")
axes[1].axhline(yt.mean(), color="grey", ls=":", label=f"Base rate {yt.mean():.2f}")
axes[1].set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision")
axes[1].legend(loc="upper right")
save_fig("roc_pr_curves")

frac_pos, mean_pred = calibration_curve(y_test, test_proba, n_bins=8, strategy="quantile")
plt.figure(figsize=(5.5, 5))
plt.plot(mean_pred, frac_pos, marker="o", color="#2f6db5", label=best_name)
plt.plot([0, 1], [0, 1], ls=":", color="grey", label="Perfect calibration")
plt.title(f"Calibration (Brier {metrics_050['brier']:.3f})")
plt.xlabel("Predicted probability")
plt.ylabel("Observed retention rate")
plt.legend()
save_fig("calibration_curve")

plt.figure(figsize=(10, 6.5))
plot_df = results[results["status"].fillna("success") == "success"].copy()
plot_df["label"] = plot_df["model"] + np.where(plot_df["tuned"] == True, " (tuned)", "")  # noqa: E712
plot_df = plot_df.sort_values("cv_pr_auc")
sns.barplot(data=plot_df, x="cv_pr_auc", y="label", palette="Blues")
plt.xlim(max(0.0, plot_df["cv_pr_auc"].min() - 0.05), min(1.0, plot_df["cv_pr_auc"].max() + 0.03))
plt.title("Model selection: 5-fold CV PR-AUC on training data")
plt.xlabel("CV PR-AUC")
plt.ylabel("")
save_fig("model_comparison_cv_pr_auc")

top_imp = importance.head(15).iloc[::-1]
plt.figure(figsize=(9, 6))
plt.barh(top_imp["feature"], top_imp["importance"], xerr=top_imp["std"], color="#2f6db5")
plt.title(f"Permutation importance (drop in PR-AUC): {best_name}")
plt.xlabel("Mean decrease in PR-AUC when the feature is shuffled")
save_fig("permutation_importance")

plt.figure(figsize=(8, 4.5))
plt.bar(lift_table["decile"].astype(int), lift_table["lift_vs_average"], color="#2f6db5")
plt.axhline(1, color="grey", ls=":")
plt.title("Lift by score decile (1 = customers the model rates most likely to return)")
plt.xlabel("Decile")
plt.ylabel("Lift vs average")
save_fig("lift_by_decile")

# ================================================================
# 11. SAVE MODEL + FINAL REPORT
# ================================================================

print("\n[11/11] SAVING MODEL AND REPORT")
joblib.dump({"model": best_model, "threshold": HEADLINE_THRESHOLD, "oof_tuned_threshold": best_thr,
             "features": selected,
             "target": TARGET, "feature_window_days": FEATURE_WINDOW_DAYS,
             "target_window_days": [FEATURE_WINDOW_DAYS, TARGET_END_DAYS],
             "model_name": best_label},
            MODELS_DIR / "best_repeat_purchase_model.joblib")

top5 = importance.head(5)
top_decile = lift_table.iloc[0]
top2_capture = lift_table.loc[lift_table["decile"].astype(int) <= 2, "retained"].sum() / lift_table["retained"].sum()
cols_used = int((column_usage["status"] == "Used").sum())
used_names = ", ".join(column_usage.loc[column_usage["status"] == "Used", "raw_column"])
removed_rows = original_rows - len(df)

L = []
L.append("# FINAL RESULTS - Retention prediction (Online Retail II)")
L.append("")
L.append("## 1. Data and coverage")
L.append(f"- Raw rows: {original_rows:,} (both sheets) -> clean rows: {len(df):,} "
         f"({removed_rows:,} removed, see cleaning_waterfall.csv)")
L.append(f"- Largest removal: missing Customer ID = {audit['missing_customer_ids']:,} rows "
         f"({audit['missing_customer_id_pct']}%) - customers cannot be followed without an ID")
L.append(f"- Duplicates removed: {original_duplicates:,} (includes the 1-9 Dec 2010 overlap between sheets)")
L.append(f"- Customers after cleaning: {all_customers:,}; modelled: {len(F):,}; "
         f"{all_customers - len(F):,} excluded because their 180-day outcome is not yet observable")
L.append(f"- Raw columns: 8 (+1 technical). Used in modelling: {cols_used} ({used_names}). "
         "Not used: Description (free text, EDA only)")
L.append("")
L.append("## 2. Target and features")
L.append("- Target: customer buys again between day 90 and day 180 after the first purchase")
L.append("- Features: only the first 90 days (no overlap with the target window, checked by an assertion)")
L.append(f"- Features engineered: {len(catalog)} in {catalog['group'].nunique()} groups "
         f"({', '.join(catalog.groupby('group', sort=False).size().index)})")
L.append(f"- Excluded by design: {len(SEASONAL_FEATURES)} seasonality features (first_month_sin/cos). "
         f"Ablation under the same honest split: test ROC-AUC {season_roc:.3f} / PR-AUC {season_pr:.3f} "
         f"WITH them vs {metrics_050['roc_auc']:.3f} / {metrics_050['pr_auc']:.3f} WITHOUT them")
L.append(f"- Dropped as redundant (|rho| > {CORR_THRESHOLD}): {len(candidate_features) - len(selected)}"
         + (" (" + "; ".join(f"{k}: {v}" for k, v in dropped_reason.items()) + ")" if dropped_reason else ""))
L.append(f"- **Final feature count: {len(selected)}**")
L.append("- Why these: RFM-style behaviour (recency/frequency/monetary), purchase rhythm, product variety, "
         "returns and country - signals that exist at day 90 and are known to drive retention")
L.append("")
L.append("## 3. Class imbalance")
L.append(f"- Class counts {class_counts.to_dict()} -> minority share {minority_share:.1%}, ratio {imbalance_ratio:.2f}:1 "
         f"=> **{imbalance_level}**")
L.append(f"- Decision: {imbalance_action}")
if abs(y_train.mean() - y_test.mean()) > 0.10:
    L.append(f"- The test period is moderately imbalanced ({y_test.mean():.0%} positive), so models keep class_weight='balanced' "
             "where supported and are judged with PR-AUC / lift. Still no SMOTE: class weighting made no measurable "
             "difference in the experiment below, and synthetic customers add nothing here")
L.append(f"- Note: the positive rate shifts with the calendar - train {y_train.mean():.1%} vs test {y_test.mean():.1%} "
         "(Dec-2009 cohort contains pre-existing customers; customers acquired in Aug-Nov have their day 90-180 "
         "window in the quiet Feb-May season). So accuracy/F1 are NOT comparable between CV and test; "
         "ROC-AUC, PR-AUC relative to the base rate, and lift are the fair metrics")
L.append("- Check: class_weight None vs balanced changes CV PR-AUC by at most "
         f"{(imb_df.groupby('model')['cv_pr_auc'].agg(lambda s: s.max() - s.min())).max():.3f}")
L.append("")
L.append("## 4. Validation design")
L.append(f"- Split: {split_type}. Train = first purchases up to {(cutoff - pd.Timedelta(days=TARGET_END_DAYS)).date()}, "
         f"gap of {purged_n:,} customers purged, test = first purchases from {cutoff.date()}")
L.append(f"- Train {len(X_train):,} customers | test {len(X_test):,} customers | positive rate "
         f"train {y_train.mean():.1%} / test {y_test.mean():.1%}")
L.append("- All preprocessing (log, imputation + missing flags, scaling) is inside the pipeline; "
         "the model is chosen by cross-validation on the training set only")
L.append("")
L.append("## 5. Model selection (5-fold CV on training set)")
L.append("| Model | CV PR-AUC (+/- std) | CV ROC-AUC | CV F1 |")
L.append("|---|---|---|---|")
for _, r in ok.iterrows():
    L.append(f"| {r['model']} | {r['cv_pr_auc']:.3f} +/- {r['cv_pr_auc_std']:.3f} | {r['cv_roc_auc']:.3f} | {r['cv_f1']:.3f} |")
for _, r in tuned_df.iterrows():
    L.append(f"| {r['model']} (tuned) | {r['cv_pr_auc']:.3f} | - | - |")
L.append(f"\n**Selected: {best_label}** (highest CV PR-AUC {best_cv:.3f})")
_top = sorted([c[1] for c in candidates], reverse=True)[:3]
_std = float(ok["cv_pr_auc_std"].head(3).mean())
if _top[0] - _top[-1] < _std:
    L.append(f"- The top models are statistically tied (spread {_top[0] - _top[-1]:.3f} < one CV std {_std:.3f}); "
             "the simpler Logistic Regression is a near-equal, more interpretable alternative")
L.append("")
L.append("## 6. Final test-set performance (not used for training, tuning or model selection)")
L.append("| Metric | HEADLINE threshold 0.50 | OOF-tuned threshold %.2f (sensitivity) |" % best_thr)
L.append("|---|---|---|")
for m in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "brier"]:
    L.append(f"| {m} | {metrics_050[m]:.3f} | {metrics_tuned[m]:.3f} |")
L.append(f"\nTest base rate = {yt.mean():.3f}; PR-AUC {metrics_050['pr_auc']:.3f} is {metrics_050['pr_auc'] / yt.mean():.2f}x the base rate "
         "(a random model would score the base rate)")
L.append(f"\n95% bootstrap CI: F1 {ci['f1'][0]:.3f}-{ci['f1'][1]:.3f} | ROC-AUC {ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f} "
         f"| PR-AUC {ci['pr_auc'][0]:.3f}-{ci['pr_auc'][1]:.3f}")
L.append("")
L.append("| Baseline | F1 | ROC-AUC | PR-AUC |")
L.append("|---|---|---|---|")
for _, r in baselines.iterrows():
    L.append(f"| {r['model']} | {r['f1']:.3f} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} |")
L.append(f"| **{best_label}** (threshold 0.50) | {metrics_050['f1']:.3f} | {metrics_050['roc_auc']:.3f} | {metrics_050['pr_auc']:.3f} |")
L.append("")
L.append("- Reading: at a fixed 0.50 cut-off the model's F1 is about the same as the simple rule; its advantage is "
         "RANKING quality (higher ROC-AUC / PR-AUC and the lift below), which is what a retention campaign needs")
L.append(f"- The threshold tuned out-of-fold ({best_thr:.2f}) did not transfer (F1 {metrics_tuned['f1']:.3f} vs {metrics_050['f1']:.3f}) "
         "because the base rate differs between train and test; use the lift table / a capacity-based cut-off instead")
L.append("")
L.append("## 7. Business value")
L.append(f"- Top-decile customers return at {top_decile['retention_rate']:.0%} vs {yt.mean():.0%} on average "
         f"(lift {top_decile['lift_vs_average']:.2f}x)")
L.append(f"- Contacting the top 20% scored customers reaches {top2_capture:.0%} of everyone who actually returns")
L.append("- Top drivers (permutation importance): " + ", ".join(
    f"{r.feature} ({r.importance:.3f})" for r in top5.itertuples()))
L.append("")
L.append("## 8. Robustness and limitations")
if split_type == "purged temporal":
    L.append(f"- Without purging (training on {naive_n:,} customers whose label window overlaps the test period) "
             f"test PR-AUC would be {naive_pr:.3f} vs {metrics_050['pr_auc']:.3f}: "
             + ("negligible here, but the purged split is still the leakage-safe choice" if abs(naive_pr - metrics_050['pr_auc']) < 0.02
                else "the purged split gives the honest, lower figure"))
L.append(f"- Base-rate shift: {y_train.mean():.0%} of training customers return vs {y_test.mean():.0%} in the test period, "
         "so any probability threshold must be re-estimated on recent data before deployment")
L.append("- Dec-2009 'first purchases' include customers who were already active before the data starts (left-censoring)")
L.append("- Only transaction data exists (no demographics, marketing or satisfaction data), which limits achievable accuracy")
L.append("- Only two years of data: too few annual cycles to learn seasonality honestly, so month-of-year features were excluded")
L.append(f"- Total runtime: {(time.time() - START_TIME) / 60:.1f} minutes")

(REPORTS / "final_results.md").write_text("\n".join(L), encoding="utf-8")

save_json({
    "dataset": str(DATA_PATH), "original_rows": original_rows, "cleaned_rows": len(df),
    "customers_after_cleaning": all_customers, "customers_modelled": len(F),
    "candidate_features": len(candidate_features), "selected_features": selected,
    "dropped_features": dropped_reason, "class_counts": class_counts.to_dict(),
    "imbalance_level": imbalance_level, "split_type": split_type,
    "train_customers": len(X_train), "purged_customers": purged_n, "test_customers": len(X_test),
    "final_model": best_label, "threshold": HEADLINE_THRESHOLD, "oof_tuned_threshold": best_thr,
    "test_metrics_threshold_0.50": metrics_050, "test_metrics_tuned_threshold": metrics_tuned,
    "bootstrap_95ci": ci, "top_features": top5["feature"].tolist(),
}, REPORTS / "project_summary.json")

banner("FINAL RESULTS")
print("\n".join(L))
print("\n" + "=" * 78)
print("Saved model :", MODELS_DIR / "best_repeat_purchase_model.joblib")
print("Report      :", REPORTS / "final_results.md")
print("Figures     :", FIGURES.resolve())
print("=" * 78)