"""
Bank Customer Churn Prediction
Dataset: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
Models: Logistic Regression (baseline) vs Random Forest
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, RocCurveDisplay)

# 1. Load data
df = pd.read_csv("Churn_Modelling.csv")
print("Shape:", df.shape)
print("Churn rate: {:.1%}".format(df["Exited"].mean()))

# 2. Clean: drop ID columns that carry no predictive information
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# 3. Encode categorical columns
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

X = df.drop(columns=["Exited"])
y = df["Exited"]

# 4. Train/test split (stratified keeps the 80/20 churn ratio in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Scale features (needed for Logistic Regression, harmless for RF)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 6. Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []
fig, ax = plt.subplots(figsize=(6, 5))
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, 1]
    acc, auc = accuracy_score(y_test, pred), roc_auc_score(y_test, prob)
    results.append({"Model": name, "Accuracy": round(acc, 3), "ROC-AUC": round(auc, 3)})
    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=ax)

ax.set_title("ROC Curve - Churn Models")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=120)

print("\n===== Summary =====")
print(pd.DataFrame(results).to_string(index=False))

# 7. Business insight: what drives churn?
rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh", figsize=(6, 5), title="Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
print("\nTop 5 churn drivers:\n", importance.sort_values(ascending=False).head(5))
