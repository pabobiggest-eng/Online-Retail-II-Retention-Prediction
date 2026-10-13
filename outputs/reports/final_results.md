# FINAL RESULTS - Retention prediction (Online Retail II)

## 1. Data and coverage
- Raw rows: 1,067,371 (both sheets) -> clean rows: 776,579 (290,792 removed, see cleaning_waterfall.csv)
- Largest removal: missing Customer ID = 243,007 rows (22.77%) - customers cannot be followed without an ID
- Duplicates removed: 34,335 (includes the 1-9 Dec 2010 overlap between sheets)
- Customers after cleaning: 5,852; modelled: 4,952; 900 excluded because their 180-day outcome is not yet observable
- Raw columns: 8 (+1 technical). Used in modelling: 7 (Invoice, StockCode, Quantity, InvoiceDate, Price, Customer ID, Country). Not used: Description (free text, EDA only)

## 2. Target and features
- Target: customer buys again between day 90 and day 180 after the first purchase
- Features: only the first 90 days (no overlap with the target window, checked by an assertion)
- Features engineered: 23 in 7 groups (Monetary, Frequency, Recency, Variety, Profile, Seasonality, Returns)
- Excluded by design: 2 seasonality features (first_month_sin/cos). Ablation under the same honest split: test ROC-AUC 0.688 / PR-AUC 0.571 WITH them vs 0.716 / 0.599 WITHOUT them
- Dropped as redundant (|rho| > 0.95): 4 (spend_per_active_day: |rho|=0.98 with average_basket_value; active_purchase_days: |rho|=0.96 with unique_invoices; spend_last_30d_share: |rho|=0.97 with invoices_last_30d; return_value_ratio: |rho|=0.98 with cancelled_invoices_90d)
- **Final feature count: 17**
- Why these: RFM-style behaviour (recency/frequency/monetary), purchase rhythm, product variety, returns and country - signals that exist at day 90 and are known to drive retention

## 3. Class imbalance
- Class counts {0: 2757, 1: 2195} -> minority share 44.3%, ratio 1.26:1 => **balanced / mild**
- Decision: No resampling needed (no SMOTE, no undersampling).
- The test period is moderately imbalanced (32% positive), so models keep class_weight='balanced' where supported and are judged with PR-AUC / lift. Still no SMOTE: class weighting made no measurable difference in the experiment below, and synthetic customers add nothing here
- Note: the positive rate shifts with the calendar - train 55.5% vs test 31.9% (Dec-2009 cohort contains pre-existing customers; customers acquired in Aug-Nov have their day 90-180 window in the quiet Feb-May season). So accuracy/F1 are NOT comparable between CV and test; ROC-AUC, PR-AUC relative to the base rate, and lift are the fair metrics
- Check: class_weight None vs balanced changes CV PR-AUC by at most 0.001

## 4. Validation design
- Split: purged temporal. Train = first purchases up to 2010-05-08, gap of 1,477 customers purged, test = first purchases from 2010-11-04
- Train 2,483 customers | test 992 customers | positive rate train 55.5% / test 31.9%
- All preprocessing (log, imputation + missing flags, scaling) is inside the pipeline; the model is chosen by cross-validation on the training set only

## 5. Model selection (5-fold CV on training set)
| Model | CV PR-AUC (+/- std) | CV ROC-AUC | CV F1 |
|---|---|---|---|
| Logistic Regression | 0.779 +/- 0.015 | 0.732 | 0.678 |
| Neural Network MLP | 0.779 +/- 0.015 | 0.730 | 0.695 |
| Extra Trees | 0.777 +/- 0.018 | 0.731 | 0.681 |
| Random Forest | 0.768 +/- 0.022 | 0.718 | 0.677 |
| Gradient Boosting | 0.767 +/- 0.022 | 0.715 | 0.684 |
| Support Vector Machine | 0.756 +/- 0.027 | 0.715 | 0.672 |
| Gaussian Naive Bayes | 0.756 +/- 0.023 | 0.719 | 0.673 |
| Hist Gradient Boosting | 0.752 +/- 0.017 | 0.698 | 0.669 |
| K-Nearest Neighbors | 0.732 +/- 0.016 | 0.704 | 0.698 |
| Decision Tree | 0.682 +/- 0.012 | 0.656 | 0.643 |
| Logistic Regression (tuned) | 0.779 | - | - |
| Neural Network MLP (tuned) | 0.778 | - | - |
| Extra Trees (tuned) | 0.781 | - | - |

**Selected: Extra Trees (tuned)** (highest CV PR-AUC 0.781)
- The top models are statistically tied (spread 0.002 < one CV std 0.016); the simpler Logistic Regression is a near-equal, more interpretable alternative

## 6. Final test-set performance (not used for training, tuning or model selection)
| Metric | HEADLINE threshold 0.50 | OOF-tuned threshold 0.30 (sensitivity) |
|---|---|---|
| accuracy | 0.707 | 0.469 |
| balanced_accuracy | 0.668 | 0.572 |
| precision | 0.538 | 0.360 |
| recall | 0.560 | 0.858 |
| f1 | 0.549 | 0.507 |
| mcc | 0.332 | 0.158 |
| roc_auc | 0.716 | 0.716 |
| pr_auc | 0.599 | 0.599 |
| brier | 0.199 | 0.199 |

Test base rate = 0.319; PR-AUC 0.599 is 1.88x the base rate (a random model would score the base rate)

95% bootstrap CI: F1 0.502-0.595 | ROC-AUC 0.679-0.751 | PR-AUC 0.548-0.652

| Baseline | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Baseline: always majority class | 0.483 | 0.500 | 0.319 |
| Baseline: rule 'bought on 2+ invoices in 90d' | 0.557 | 0.688 | 0.501 |
| **Extra Trees (tuned)** (threshold 0.50) | 0.549 | 0.716 | 0.599 |

- Reading: at a fixed 0.50 cut-off the model's F1 is about the same as the simple rule; its advantage is RANKING quality (higher ROC-AUC / PR-AUC and the lift below), which is what a retention campaign needs
- The threshold tuned out-of-fold (0.30) did not transfer (F1 0.507 vs 0.549) because the base rate differs between train and test; use the lift table / a capacity-based cut-off instead

## 7. Business value
- Top-decile customers return at 75% vs 32% on average (lift 2.35x)
- Contacting the top 20% scored customers reaches 42% of everyone who actually returns
- Top drivers (permutation importance): average_days_between_visits (0.035), days_to_second_purchase (0.033), recency_days_at_day_90 (0.012), total_spend (0.005), unique_invoices (0.004)

## 8. Robustness and limitations
- Without purging (training on 3,960 customers whose label window overlaps the test period) test PR-AUC would be 0.601 vs 0.599: negligible here, but the purged split is still the leakage-safe choice
- Base-rate shift: 55% of training customers return vs 32% in the test period, so any probability threshold must be re-estimated on recent data before deployment
- Dec-2009 'first purchases' include customers who were already active before the data starts (left-censoring)
- Only transaction data exists (no demographics, marketing or satisfaction data), which limits achievable accuracy
- Only two years of data: too few annual cycles to learn seasonality honestly, so month-of-year features were excluded
- Total runtime: 4.1 minutes