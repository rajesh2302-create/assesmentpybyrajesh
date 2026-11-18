
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 0. Dataset Setup
# ---------------------------------------------------------
def get_loan_data(n=1500, random_state=42):
    rng = np.random.RandomState(random_state)
    income = rng.lognormal(mean=10.8, sigma=0.6, size=n)
    credit_score = rng.normal(loc=650, scale=75, size=n).clip(300, 850)
    age = rng.randint(21, 70, size=n)
    years_employed = rng.randint(0, 30, size=n)
    noise_feat1 = rng.normal(0, 1, size=n)
    noise_feat2 = rng.normal(0, 1, size=n)
    
    # Loan amount depends on income and credit score
    loan_amount = 0.25 * income + 15 * credit_score + rng.normal(0, 3000, size=n)
    loan_amount = loan_amount.clip(1000, None)
    
    dti = loan_amount / (income + 1)
    prob = 1 / (1 + np.exp(-(-2.0 + 3.0 * dti - 0.008 * (credit_score - 600))))
    default = (prob > 0.5).astype(int)
    
    return pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'age': age,
        'years_employed': years_employed,
        'noise_feat1': noise_feat1,
        'noise_feat2': noise_feat2,
        'loan_amount': loan_amount,
        'default': default
    })

df = get_loan_data()
feature_cols = ['income', 'credit_score', 'age', 'years_employed', 'noise_feat1', 'noise_feat2', 'loan_amount']
X = df[feature_cols]
y = df['default']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# ---------------------------------------------------------
# 1. Train 4 Classifiers (LogReg, DecisionTree, RandomForest, XGBoost)
# ---------------------------------------------------------
pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42))])
pipe_dt = Pipeline([('scaler', StandardScaler()), ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))])
pipe_rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=150, oob_score=True, random_state=42))])

pipe_lr.fit(X_train, y_train)
pipe_dt.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

# XGBoost with Early Stopping on Validation Set
scaler_xgb = StandardScaler()
X_train_xgb = scaler_xgb.fit_transform(X_train)
X_val_xgb = scaler_xgb.transform(X_val)

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    early_stopping_rounds=20,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)

# Print Random Forest OOB Score
print(f"Random Forest OOB Score: {pipe_rf.named_steps['clf'].oob_score_:.4f}")

# Results Table: Model | Validation Score
results = [
    ("Logistic Regression", accuracy_score(y_val, pipe_lr.predict(X_val))),
    ("Decision Tree (depth=5)", accuracy_score(y_val, pipe_dt.predict(X_val))),
    ("Random Forest", accuracy_score(y_val, pipe_rf.predict(X_val))),
    ("XGBoost (Early Stopping)", accuracy_score(y_val, xgb_model.predict(X_val_xgb)))
]

print("\n--- Classification Results Table ---")
print(f"{'Model':<28} | {'Validation Score':<16}")
print("-" * 47)
for name, score in results:
    print(f"{name:<28} | {score:.4f}")

# ---------------------------------------------------------
# 2. Compare predict_proba vs predict for 5 rows (Logistic Regression)
# ---------------------------------------------------------
probs_5 = pipe_lr.predict_proba(X_val.iloc[:5])
preds_5 = pipe_lr.predict(X_val.iloc[:5])
print("\n--- Logistic Regression: predict_proba vs predict (First 5 Val Rows) ---")
for i in range(5):
    print(f"Row {i+1}: P(0)={probs_5[i][0]:.3f}, P(1)={probs_5[i][1]:.3f}  --> Predicted Class: {preds_5[i]}")

# ---------------------------------------------------------
# 3. Top 5 Most Important Features (Random Forest & XGBoost)
# ---------------------------------------------------------
rf_imp = pd.Series(pipe_rf.named_steps['clf'].feature_importances_, index=feature_cols).sort_values(ascending=False)
xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

print("\nTop 5 Features (Random Forest):")
print(rf_imp.head(5).round(4))

print("\nTop 5 Features (XGBoost):")
print(xgb_imp.head(5).round(4))

# ---------------------------------------------------------
# 4. Regression: Linear, Ridge, Lasso to predict 'loan_amount'
# ---------------------------------------------------------
reg_features = ['income', 'credit_score', 'age', 'years_employed', 'noise_feat1', 'noise_feat2']
Xr_train, Xr_val = X_train[reg_features], X_val[reg_features]
yr_train, yr_val = X_train['loan_amount'], X_val['loan_amount']

scaler_reg = StandardScaler()
Xr_tr_s = scaler_reg.fit_transform(Xr_train)
Xr_va_s = scaler_reg.transform(Xr_val)

lin_reg = LinearRegression().fit(Xr_tr_s, yr_train)
ridge_reg = Ridge(alpha=10.0).fit(Xr_tr_s, yr_train)
lasso_reg = Lasso(alpha=500.0).fit(Xr_tr_s, yr_train)

coef_df = pd.DataFrame({
    'Feature': reg_features,
    'Linear': lin_reg.coef_,
    'Ridge (alpha=10)': ridge_reg.coef_,
    'Lasso (alpha=500)': lasso_reg.coef_
})
print("\n--- Regression Coefficient Comparison (Target: loan_amount) ---")
print(coef_df.round(2).to_string(index=False))
zero_coefs = np.sum(np.isclose(lasso_reg.coef_, 0))
print(f"Lasso Zero Coefficients Count: {zero_coefs} out of {len(reg_features)}")

# ---------------------------------------------------------
# 5. Unsupervised: K-Means Clustering (k = 2 to 8)
# ---------------------------------------------------------
cluster_cols = ['income', 'loan_amount', 'credit_score']
X_clust = StandardScaler().fit_transform(df[cluster_cols])

k_values = range(2, 9)
inertias = []
sil_scores = []

print("\n--- K-Means Clustering (k = 2 to 8) ---")
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clust)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_clust, labels)
    sil_scores.append(sil)
    print(f"k = {k} | Inertia: {km.inertia_:8.1f} | Silhouette Score: {sil:.4f}")

# Plot Elbow Curve and Silhouette Scores
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(k_values, inertias, marker='o')
ax[0].set_title("K-Means Elbow Curve")
ax[0].set_xlabel("Number of Clusters (k)")
ax[0].set_ylabel("Inertia")
ax[0].grid(True)

ax[1].plot(k_values, sil_scores, marker='s', color='green')
ax[1].set_title("Silhouette Scores by k")
ax[1].set_xlabel("Number of Clusters (k)")
ax[1].set_ylabel("Silhouette Score")
ax[1].grid(True)

plt.tight_layout()
plt.savefig("task3_kmeans_elbow_silhouette.png")
print("Saved plot: task3_kmeans_elbow_silhouette.png")

# Describe segments for k=3
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Segment'] = km3.fit_predict(X_clust)
print("\nCustomer Segment Profiles (Mean values for k=3):")
print(df.groupby('Segment')[cluster_cols].mean().round(1))

