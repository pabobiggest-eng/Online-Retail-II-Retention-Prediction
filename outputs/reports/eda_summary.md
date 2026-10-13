# EDA Summary - Online Retail II

- Period: 2009-12-01 to 2011-12-09 (776,579 transaction rows, 5,852 identified customers, 36,594 invoices).
- Total revenue: GBP 17,068,582.72; UK revenue share: 83.7%.
- Average order value: GBP 466.43; median order value: GBP 302.55.
- 72.3% of identified customers placed at least two invoices.
- Repeat customers account for 96.7% of identified-customer revenue.
- The top 20% of identified customers account for 77.2% of identified-customer revenue.
- Highest-revenue full month: 2010-11 (GBP 1,155,978.39).
- Highest-revenue product: REGENCY CAKESTAND 3 TIER (StockCode 22423; GBP 277,656.25).
- Average month-1 cohort retention: 21.0%.
- Orders on Saturday: 0.1% of all invoices.
- NOTE: 2011-12 may be incomplete because the latest transaction date is 2011-12-09; it is excluded from the full-month peak comparison.

## Interpretation notes
- Customer-level metrics exclude transactions/invoices without an identifiable Customer ID; transaction and revenue totals include all usable cleaned rows.
- Cohort retention compares each customer's first observed purchase month with later observed active months. The first cohort can be affected by customers whose earlier history predates this dataset.
- This EDA reads the existing cleaned CSV; cleaning rules are applied by the main project pipeline, not re-created here.
