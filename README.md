# Telco Customer Churn — Recall-Optimized Prediction

A recall-optimized churn classification project. Identifies telecom customers likely to leave by combining systematic EDA, business cost analysis, and interpretable modeling. Achieves **82% recall** — catching 4 out of 5 churners before they leave.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikitlearn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

---

## Business Problem

Customer churn costs telecom companies 5–10× more than retention. The cost structure is asymmetric:

```
Cost of missing a churner (False Negative):  ~$840/year lost recurring revenue
Cost of a false alarm (False Positive):      ~$20 unnecessary retention offer

Cost ratio: 42:1 → optimize for recall, not accuracy
```

A naive majority-class classifier achieves 73.5% accuracy by predicting "no churn" for everyone — while catching zero churners. **Accuracy is the wrong metric for this problem.**

---

## Key Findings

### High-Risk Segment Profile

| Risk Factor | Churn Rate | vs. Baseline (26.5%) | Ratio |
|-------------|-----------|---------------------|-------|
| Month-to-month contract | 42.7% | +61% | 15.3× vs two-year |
| Electronic check payment | 45.3% | +71% | 3.0× vs autopay |
| Tenure < 6 months | 53.0% | +100% | 2.0× vs overall |
| Fiber optic internet | 41.9% | +58% | 2.2× vs DSL |
| No tech support | ~40% | +51% | 2.8× vs with support |

> **The highest-risk intersection:** New customers (< 6 months), on month-to-month contracts, using fiber optic without tech support, paying by electronic check. This segment churns at **~55%** — more than double the baseline.

### Convergent Validation

The same risk factors emerged independently from three different methods:
1. **Univariate EDA** — churn rates by category
2. **Logistic Regression coefficients** — standardized feature weights
3. **Random Forest importance** — Gini-based feature ranking

This cross-methodology consistency provides strong confidence that these are genuine predictors, not statistical artifacts.

---

## Model Performance

| Model | Accuracy | Recall | Precision | F1 | Strategy |
|-------|----------|--------|-----------|-----|----------|
| Logistic Regression (baseline) | 0.807 | 0.567 | 0.658 | 0.609 | Default threshold |
| Logistic Regression (balanced) | 0.740 | 0.786 | 0.507 | 0.616 | Class weights |
| Random Forest (full depth, balanced) | 0.790 | 0.495 | 0.634 | 0.556 | Weights negated by depth |
| **Random Forest (depth=5, balanced)** | **0.737** | **0.816** | **0.502** | **0.622** | **Optimal** |

**Selected model:** Class-weighted Random Forest with `max_depth=5`.

**Key modeling insight:** Unlimited tree depth *negates* class weights. Pure leaves have zero loss regardless of weight multiplier, so the forest reverts to majority-class behavior (recall drops to 0.50). Limiting depth to 5 keeps leaves impure, allowing weighted loss to shift the decision boundary toward catching more churners.

---

## Methodology

```
Phase 1: Data Assessment       →  7,043 records, 21 features, 26.5% churn rate
Phase 2: Data Quality          →  11 missing TotalCharges (tenure=0 → imputed as $0)
Phase 3: Systematic EDA        →  Demographics → Services → Contract/Billing → Numerics
Phase 4: Risk Segmentation     →  Identified high-risk intersection, quantified business impact
Phase 5: Preprocessing         →  One-hot encoding, stratified split, no SMOTE needed
Phase 6: Baseline Modeling     →  LogReg (interpretable) → coefficient analysis
Phase 7: Class Imbalance       →  class_weight="balanced" → +38% recall improvement
Phase 8: Hyperparameter Study  →  max_depth × class_weight interaction discovered
Phase 9: Feature Importance    →  Convergent validation across 3 methodologies
```

### Why Not SMOTE?

| Approach | When to Use | Why Not Here |
|----------|------------|-------------|
| SMOTE | Severe imbalance (<5% minority), continuous features | 26.5% is moderate; mostly categorical features → synthetic samples unrealistic |
| Class weights | Moderate imbalance, any feature type | ✓ Clean, no synthetic data, works with loss function |
| Threshold tuning | Need calibrated probabilities | Valid alternative — planned for Phase 2 |

---

## Actionable Recommendations

| Priority | Action | Target Segment | Expected Impact |
|----------|--------|----------------|-----------------|
| 🔴 Critical | 12-month contract incentive during onboarding | New customers (< 3 mo) | Reduce M2M exposure |
| 🔴 Critical | Bundle free tech support with fiber plans | All fiber customers | Address 42% churn |
| 🟡 High | 5% autopay discount for e-check switchers | Electronic check users | Reduce payment friction |
| 🟡 High | Proactive outreach at month 3–4 | All new customers | Intervene before 6-month cliff |
| 🟢 Medium | Senior-specific retention program | Senior citizens | Address 41.7% churn |

---

## Future Work

### Phase 2: Advanced Modeling (Planned)
- **Gradient boosting:** XGBoost, LightGBM, CatBoost — expected to outperform RF on tabular data
- **Optuna hyperparameter search:** 100+ trials optimizing recall at precision ≥ 0.45
- **Stratified 5-fold CV:** Robust performance estimates (current results are single split)

### Phase 3: Threshold Tuning & Cost-Sensitive Evaluation
- Precision-Recall curve analysis
- Expected Value Framework: sweep thresholds, plot net business value
- Select optimal operating point based on campaign budget constraints

### Phase 4: Model Explainability (SHAP)
- Global and local SHAP values for per-customer explanations
- Interaction effects (tenure × contract, fiber × tech support)
- Fairness audit across demographic groups

### Phase 5: Production Considerations
- Drift monitoring and retraining cadence
- A/B testing of model-driven vs. rule-based retention campaigns
- Batch vs. event-triggered scoring pipeline design

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas / NumPy | Data manipulation |
| Seaborn / Matplotlib | Visualization |
| Scikit-learn | LogReg, Random Forest, evaluation metrics |
| Kaggle Notebooks | Development environment |

---

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM-sourced dataset with 7,043 customer records and 21 features covering demographics, service subscriptions, account details, and churn labels.

---

## Project Structure

```
telco-churn-prediction/
├── telco-customer-churn-eda-recall-optimized-predi.ipynb   # Complete analysis (58 cells)
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/abdullahaltepe-hue/telco-churn-prediction.git
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook telco-customer-churn-eda-recall-optimized-predi.ipynb
```

Or view directly on [Kaggle](https://www.kaggle.com/abdullahaltepe).

---

## Limitations

| Limitation | Impact | Planned Mitigation |
|-----------|--------|-------------------|
| Static snapshot (no temporal dimension) | Cannot capture behavioral trends | Event-based features in Phase 5 |
| Single train/test split | Performance estimate has variance | 5-fold CV in Phase 2 |
| No external validation | May not generalize to other telcos | Patterns are industry-universal |
| Two model families tested | Potentially suboptimal | Gradient boosting in Phase 2 |

---

*Built by [Abdullah Altepe](https://github.com/abdullahaltepe-hue) | [LinkedIn](https://www.linkedin.com/in/abdullahaltepe)*
