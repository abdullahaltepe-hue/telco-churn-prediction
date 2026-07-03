# Telco Customer Churn Prediction

Predicting churn for 7,043 telecom customers.
Final model catches **82% of churners** (recall 0.82).

## Highlights
- EDA: high-risk profile = new, month-to-month fiber customers without tech support, paying by electronic check
- Class imbalance handled with class weights; metric choice driven by business costs (missed churner ≈ 40x a false alarm)
- Model comparison: Logistic Regression vs Random Forest + hyperparameter analysis

## Links
- [Notebook on Kaggle](https://www.kaggle.com/abdullahaltepe)

## Tools
Python · pandas · seaborn · scikit-learn
