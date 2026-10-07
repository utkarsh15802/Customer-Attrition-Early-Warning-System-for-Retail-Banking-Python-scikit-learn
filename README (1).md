# Bank Customer Churn Prediction

Predicts which bank customers are likely to leave, using Logistic Regression (baseline) and Random Forest, and identifies the main drivers of churn to guide retention strategy.

## Business problem
Acquiring a new customer costs several times more than retaining one. Flagging high-risk customers early lets a bank target retention offers where they matter.

## Dataset
[Churn Modelling – Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling): 10,000 customers, 14 columns. Target: `Exited` (1 = left the bank). About 20% of customers churned.

## Approach
1. Dropped ID columns (`RowNumber`, `CustomerId`, `Surname`)
2. One-hot encoded `Geography` and `Gender`
3. Stratified 80/20 train-test split
4. Standardised features
5. Trained Logistic Regression and Random Forest, compared on Accuracy and ROC-AUC
6. Used Random Forest feature importance to explain churn drivers

## Results
| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | _fill after run_ | _fill after run_ |
| Random Forest | _fill after run_ | _fill after run_ |

![ROC Curve](roc_curve.png)
![Feature Importance](feature_importance.png)

## Key insights
_Fill in from the top 5 churn drivers printed by the script, e.g. age, number of products, activity status._

## How to run
```bash
pip install -r requirements.txt
# download Churn_Modelling.csv from Kaggle into this folder
python churn_prediction.py
```
