import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    mean_absolute_error, root_mean_squared_error, r2_score
)

# ---------------------------------------------------------
# 0. Dataset Setup (Imbalanced ~20% Default Rate)
# ---------------------------------------------------------
def get_loan_data(n=1500, random_state=42):
    rng = np.random.RandomState(random_state)
    income = rng.lognormal(mean=10.8, sigma=0.6, size=n)
    credit_score = rng.normal(loc=650, scale=75, size=n).clip(300, 850)
    age = rng.randint(21, 70, size=n)
    loan_amount = (0.25 * income + 15 * credit_score + rng.normal(0, 3000, size=n)).clip(1000, None)
    
    dti = loan_amount / (income + 1)
    prob = 1 / (1 + np.exp(-(-2.4 + 3.0 * dti - 0.008 * (credit_score - 600))))
    default = (prob > 0.5).astype(int)
    
    return pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'age': age,
        'loan_amount': loan_amount,
        'default': default
    })

df = get_loan_data()
X = df[['income', 'credit_score', 'age', 'loan_amount']]
y = df['default']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------
# 1. Evaluate 3 Classifiers (Metrics & Confusion Matrix)
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
}

print("--- Validation Classification Metrics ---")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_val_s)
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    cm = confusion_matrix(y_val, preds)
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

print("\nWhy Accuracy Misleads on Imbalanced Data:")
print("If 85% of loans do not default, a dummy model predicting '0' for everyone gets 85% accuracy while catching 0% of actual defaulters.")

# ---------------------------------------------------------
# 2. Plot ROC Curves of 3 Models in One Chart with AUC
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
for name, model in models.items():
    probs = model.predict_proba(X_val_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, probs)
    auc_score = roc_auc_score(y_val, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.500)')
plt.title("ROC Curves Comparison (Validation Set)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("task4_roc_curves.png")
print("\nSaved plot: task4_roc_curves.png")

# ---------------------------------------------------------
# 3. Decision Threshold Tuning for >= 75% Recall
# ---------------------------------------------------------
xgb_probs = models["XGBoost"].predict_proba(X_val_s)[:, 1]
thresholds = np.arange(0.05, 0.95, 0.01)

chosen_thresh = 0.50
chosen_prec = 0.0
chosen_rec = 0.0

# Search from highest threshold down so we keep precision as high as possible while recall >= 0.75
for t in sorted(thresholds, reverse=True):
    preds_t = (xgb_probs >= t).astype(int)
    rec_t = recall_score(y_val, preds_t)
    if rec_t >= 0.75:
        chosen_thresh = t
        chosen_rec = rec_t
        chosen_prec = precision_score(y_val, preds_t)
        break

print(f"\n--- Threshold Tuning (XGBoost) ---")
print(f"Threshold for >= 75% Recall: {chosen_thresh:.2f} | Recall: {chosen_rec:.4f} | Precision: {chosen_prec:.4f}")

# ---------------------------------------------------------
# 4. Hyperparameter Tuning: GridSearchCV (RF) & RandomizedSearchCV (XGB)
# ---------------------------------------------------------
# GridSearchCV on Random Forest (3 parameters)
rf_param_grid = {
    'n_estimators':[50,100],
    'max_depth': [4, 8, None],
    'min_samples_split': [2, 10]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train_s, y_train)
print(f"\nGridSearchCV Random Forest Best Params: {grid_rf.best_params_}")
print(f"GridSearchCV Random Forest Best CV AUC: {grid_rf.best_score_:.4f}")

# RandomizedSearchCV on XGBoost (n_iter=15)
xgb_param_dist = {
    'n_estimators':[50,100],
    'max_depth':[4,8,None],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
rand_xgb = RandomizedSearchCV(
    XGBClassifier(random_state=42),
    param_distributions=xgb_param_dist,
    n_iter=15,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)
rand_xgb.fit(X_train_s, y_train)
print(f"\nRandomizedSearchCV XGBoost Best Params: {rand_xgb.best_params_}")
print(f"RandomizedSearchCV XGBoost Best CV AUC: {rand_xgb.best_score_:.4f}")

# ---------------------------------------------------------
# 5. Regression Metrics vs Mean Baseline & Final Test Evaluation
# ---------------------------------------------------------
# Regression task: predict 'loan_amount'
Xr_tr, Xr_te = X_train[['income', 'credit_score', 'age']], X_test[['income', 'credit_score', 'age']]
yr_tr, yr_te = X_train['loan_amount'], X_test['loan_amount']

reg_sc = StandardScaler()
Xr_tr_s = reg_sc.fit_transform(Xr_tr)
Xr_te_s = reg_sc.transform(Xr_te)

ridge = Ridge(alpha=1.0).fit(Xr_tr_s, yr_tr)
reg_preds = ridge.predict(Xr_te_s)
baseline_preds = np.full_like(yr_te, yr_tr.mean())

print("\n--- Regression Evaluation vs Mean Baseline (Test Set) ---")
print(f"Ridge Model   -> MAE: {mean_absolute_error(yr_te, reg_preds):.2f} | RMSE: {root_mean_squared_error(yr_te, reg_preds):.2f} | R2: {r2_score(yr_te, reg_preds):.4f}")
print(f"Mean Baseline -> MAE: {mean_absolute_error(yr_te, baseline_preds):.2f} | RMSE: {root_mean_squared_error(yr_te, baseline_preds):.2f} | R2: {r2_score(yr_te, baseline_preds):.4f}")

# Final Single Evaluation of Best Classifier (Tuned XGBoost) on Unseen Test Set
best_model = rand_xgb.best_estimator_
test_probs = best_model.predict_proba(X_test_s)[:, 1]
test_preds = best_model.predict(X_test_s)

test_auc = roc_auc_score(y_test, test_probs)
test_f1 = f1_score(y_test, test_preds)

print("\n--- Final Unseen Test Set Evaluation ---")
print(f"Best model: Tuned XGBoost | ROC-AUC: {test_auc:.2f} | F1: {test_f1:.2f}")
