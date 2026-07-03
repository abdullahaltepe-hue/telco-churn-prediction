# Telco Customer Churn Prediction

A recall-optimized classification project that identifies telecom customers likely to churn. Analyzed 7,043 customers, built interpretable models, and achieved **82% recall** — catching 4 out of 5 churners before they leave.

## Business Problem

Customer churn costs telecom companies 5–10x more than retention. A missed churner represents ~$840/year in lost revenue, while a false alarm (unnecessary retention offer) costs only ~$20. This asymmetry demands a **recall-first** modeling strategy.

**Goal**: Build a model that maximizes churner detection (recall) while maintaining acceptable precision, enabling proactive retention campaigns.

## Key Findings

### High-Risk Customer Profile

| Risk Factor | Churn Rate | vs. Baseline (26.5%) |
|-------------|-----------|---------------------|
| Month-to-month contract | 43% | +62% |
| Fiber optic internet | 42% | +58% |
| No tech support | 42% | +58% |
| Electronic check payment | 45% | +70% |
| Tenure < 6 months | 53% | +100% |

> The highest-risk segment: **new customers (< 6 months), on month-to-month contracts, with fiber optic internet, no tech support, paying by electronic check.**

### Model Performance

| Model | Accuracy | Recall | Precision |
|-------|----------|--------|-----------|
| Logistic Regression (baseline) | 0.807 | 0.567 | 0.658 |
| Logistic Regression (balanced) | 0.740 | 0.786 | 0.507 |
| Random Forest (300 trees, full depth) | 0.790 | 0.495 | 0.634 |
| **Random Forest (300 trees, max_depth=5, balanced)** | **0.780** | **0.820** | **0.500** |

**Selected model**: Class-weighted Random Forest with `max_depth=5`. Shallow trees allow class weights to function properly — unlimited depth grows pure leaves, negating the rebalancing effect.

## Methodology

```
1. Data Loading & Inspection    →  7,043 records, 21 features
2. Data Cleaning                →  11 missing TotalCharges (tenure=0, filled with 0)
3. Exploratory Data Analysis    →  Contract, tenure, internet type, support, payment method
4. Feature Engineering          →  One-hot encoding, tenure/charge binning for EDA
5. Modeling                     →  Logistic Regression → Random Forest (hyperparameter analysis)
6. Evaluation                   →  Recall-optimized with business cost justification
```

### Why Recall Over Accuracy?

```
Cost of missing a churner (False Negative):  ~$840/year lost revenue
Cost of a false alarm (False Positive):      ~$20 retention offer

Cost ratio: 42:1 → optimize for recall
```

## Actionable Recommendations

1. **Target early tenure customers** — 53% churn in first 6 months; onboarding intervention is critical
2. **Incentivize long-term contracts** — 2-year contract churn is only 3% vs. 43% month-to-month
3. **Bundle tech support with fiber** — Fiber without support has 42% churn
4. **Migrate electronic check payers** — Incentivize switch to automatic payments (45% → 15% churn)

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas | Data manipulation |
| Seaborn / Matplotlib | Visualization |
| Scikit-learn | Modeling (LogReg, Random Forest, pipelines) |
| Kaggle Notebooks | Development environment |

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM sample dataset with 7,043 records and 21 features covering demographics, account information, and service subscriptions.

## Project Structure

```
├── telco-customer-churn-eda-recall-optimized-predi.ipynb   # Full analysis notebook
└── README.md
```

## How to Run

```bash
# Clone the repository
git clone https://github.com/abdullahaltepe-hue/telco-churn-prediction.git

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Open the notebook
jupyter notebook telco-customer-churn-eda-recall-optimized-predi.ipynb
```

Or view directly on [Kaggle](https://www.kaggle.com/abdullahaltepe).
