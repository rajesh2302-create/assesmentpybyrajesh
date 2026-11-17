import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. Create imbalanced dataset & split
X, y = make_classification(
    n_samples=1500, weights=[0.90, 0.10], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 2. Train models & get predict_proba
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
probas = {}

print(f"{'Model':<22} | {'AUC':<6} | {'F1':<6}")
print("-" * 40)

for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    probas[name] = proba

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, model.predict(X_test))
    print(f"{name:<22} | {auc:.3f}  | {f1:.3f}")

    # Plot ROC & Precision-Recall curves
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax2.plot(rec, prec, label=name)

# Format plots
ax1.plot([0, 1], [0, 1], "k--", label="Random")
ax1.set(
    title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate"
)
ax1.legend()

ax2.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
ax2.legend()
plt.tight_layout()
plt.savefig("roc_pr_curves.png")
plt.show()


# 3. Bonus Challenge: Function to find threshold with highest precision for min_recall
def best_threshold(y_true, proba, min_recall):
    prec, rec, thresh = precision_recall_curve(y_true, proba)
    valid = np.where(rec[:-1] >= min_recall)[0]
    best_idx = valid[np.argmax(prec[valid])]
    return thresh[best_idx], prec[best_idx], rec[best_idx]


# 4. Find threshold for >= 80% recall on XGBoost
thresh, prec, rec = best_threshold(y_test, probas["XGBoost"], min_recall=0.80)
print(
    f"\nXGBoost (>=80% Recall) -> Threshold: {thresh:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}"
)
