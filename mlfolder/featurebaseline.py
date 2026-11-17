import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Sample Churn Dataset (Offline - Zero Internet Error)
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'Contract': np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], n
    ),
    'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], n),
    'OnlineSecurity': np.random.choice(['Yes', 'No'], n),
    'TechSupport': np.random.choice(['Yes', 'No'], n),
    'tenure': np.random.randint(1, 72, n),
    'MonthlyCharges': np.round(np.random.uniform(20, 115, n), 2),
    'TotalCharges': np.round(np.random.uniform(100, 8000, n), 2),
    'Churn': np.random.choice([0, 1], n, p=[0.73, 0.27]),
})

# 2. Feature Engineering: Create 4 New Features
df['tenure_group'] = pd.cut(
    df['tenure'], bins=[0, 12, 48, 100], labels=['New', 'Mid', 'Loyal']
).astype(str)
df['spend_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
df['service_count'] = (df['OnlineSecurity'] == 'Yes').astype(int) + (
    df['TechSupport'] == 'Yes'
).astype(int)
df['is_monthly_contract'] = (df['Contract'] == 'Month-to-month').astype(int)

# 3. Train-Test Split (Hold out 20% test set)
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Build ColumnTransformer + Preprocessing Pipeline
num_cols = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'spend_per_month',
    'service_count',
    'is_monthly_contract',
]
cat_cols = [
    'Contract',
    'InternetService',
    'OnlineSecurity',
    'TechSupport',
    'tenure_group',
]

prep = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer()), ('sc', StandardScaler())]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])

# 5. Train & Evaluate Baseline Models with 5-Fold Cross-Validation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
}

print(f"{'Model':<22} | {'CV ROC-AUC':<10} | {'CV F1':<8}")
print('-' * 46)

for name, clf in models.items():
  pipe = Pipeline([('prep', prep), ('clf', clf)])
  scores = cross_validate(
      pipe, X_train, y_train, cv=5, scoring=['roc_auc', 'f1']
  )
  print(
      f"{name:<22} | {scores['test_roc_auc'].mean():<10.2f} |"
      f" {scores['test_f1'].mean():<8.2f}"
  )
