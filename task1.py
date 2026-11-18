<<<<<<< HEAD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 0. Dataset Creation (Replace with pd.read_csv('loan.csv') if provided)
# ---------------------------------------------------------
def get_loan_data(n=1500, random_state=42):
    rng = np.random.RandomState(random_state)
    income = rng.lognormal(mean=10.8, sigma=0.6, size=n)
    loan_amount = rng.lognormal(mean=9.5, sigma=0.7, size=n)
    credit_score = rng.normal(loc=650, scale=75, size=n).clip(300, 850)
    age = rng.randint(21, 70, size=n)
    dti = (loan_amount / (income + 1)) * 100
    
    # Create target with slight class imbalance (~25% default)
    logit = -2.2 + 0.04 * dti - 0.008 * (credit_score - 600) + rng.normal(0, 0.8, size=n)
    prob = 1 / (1 + np.exp(-logit))
    default = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'income': income,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'age': age,
        'dti': dti,
        'default': default
    })
    return df

df = get_loan_data()
X = df.drop(columns=['default'])
y = df['default']

# ---------------------------------------------------------
# 1. Train / Validation / Test Split (60 / 20 / 20 Stratified)
# ---------------------------------------------------------
# First split: 60% Train, 40% Temp (Val + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
# Second split: Split 40% Temp equally into 20% Val and 20% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Dataset Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape} (Test set put aside)")
print("-" * 65)

# ---------------------------------------------------------
# 2. Show Overfitting with DecisionTreeClassifier
# ---------------------------------------------------------
depths = [1, 3, 5, 10, None]
train_accs = []
val_accs = []

print("Decision Tree Depth Analysis (Overfitting Check):")
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    
    tr_acc = accuracy_score(y_train, dt.predict(X_train))
    va_acc = accuracy_score(y_val, dt.predict(X_val))
    
    train_accs.append(tr_acc)
    val_accs.append(va_acc)
    print(f"Depth {str(d):<4}: train {tr_acc:.2f} | val {va_acc:.2f}")

# Plot Train vs Validation Accuracy
plt.figure(figsize=(8, 4))
depth_labels = [str(d) for d in depths]
plt.plot(depth_labels, train_accs, marker='o', label='Train Accuracy')
plt.plot(depth_labels, val_accs, marker='s', label='Validation Accuracy')
plt.title("Overfitting Demonstration: Decision Tree Max Depth")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("task1_overfitting_plot.png")
print("Saved plot: task1_overfitting_plot.png")
print("-" * 65)

# ---------------------------------------------------------
# 3. 5-Fold Cross-Validation on LogisticRegression
# ---------------------------------------------------------
# Scale training data properly before CV (or use pipeline)
scaler_cv = StandardScaler()
X_train_scaled_cv = scaler_cv.fit_transform(X_train)

lr = LogisticRegression(random_state=42)
cv_scores = cross_val_score(lr, X_train_scaled_cv, y_train, cv=5, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")
print("-" * 65)

# ---------------------------------------------------------
# 4. Data Leakage Demonstration
# ---------------------------------------------------------
# WRONG WAY (Leakage): Fit scaler on ALL data (Train + Val) before splitting
scaler_leak = StandardScaler()
X_all_tv = pd.concat([X_train, X_val])
y_all_tv = pd.concat([y_train, y_val])
X_leaked_scaled = scaler_leak.fit_transform(X_all_tv)

X_tr_leak = X_leaked_scaled[:len(X_train)]
X_va_leak = X_leaked_scaled[len(X_train):]

lr_leak = LogisticRegression(random_state=42).fit(X_tr_leak, y_train)
leak_val_acc = accuracy_score(y_val, lr_leak.predict(X_va_leak))

# RIGHT WAY (No Leakage): Fit scaler ONLY on Train, then transform Val
scaler_clean = StandardScaler()
X_tr_clean = scaler_clean.fit_transform(X_train)
X_va_clean = scaler_clean.transform(X_val)

lr_clean = LogisticRegression(random_state=42).fit(X_tr_clean, y_train)
clean_val_acc = accuracy_score(y_val, lr_clean.predict(X_va_clean))

print(f"Data Leakage Demo -> Leaky Val Acc: {leak_val_acc:.4f} | Clean Val Acc: {clean_val_acc:.4f}")
print("""
Explanation (4 lines):
1. Scaling before splitting leaks the mean and standard deviation of the validation set into training.
2. The model gets unfair knowledge about the distribution of unseen data, causing optimistic bias.
3. Fitting the scaler only on X_train ensures validation data remains strictly unseen during training.
4. In production, future data will arrive one row at a time, so preprocessing must rely only on training stats.
""")
print("-" * 65)

# ---------------------------------------------------------
# 5. Problem Classification & 7-Step ML Workflow
# ---------------------------------------------------------
print("Problem Classification:")
print("1. Spam detection         -> Supervised Learning (Binary Classification)")
print("2. Customer segmentation  -> Unsupervised Learning (Clustering)")
print("3. House price prediction -> Supervised Learning (Regression)")
print("4. Anomaly detection      -> Unsupervised Learning (Outlier Detection)")

print("\n7-Step Machine Learning Workflow:")
steps = [
    "1. Problem Definition & Data Collection",
    "2. Exploratory Data Analysis (EDA) & Data Splitting (Train/Val/Test)",
    "3. Data Cleaning & Preprocessing (Imputation, Encoding, Scaling)",
    "4. Feature Engineering & Selection",
    "5. Model Training & Baseline Building",
    "6. Model Evaluation & Hyperparameter Tuning (using Validation/CV)",
    "7. Final Testing on Unseen Test Set & Deployment"
]
for s in steps:
    print(s)
=======
sales=[12000,0,8500,15000,-200,9800]
total=0
for x in sales:
    if x <= 0:
        print(f"skipped invalid value {x}")

        continue
    total+=x
    print(f"added{x}total{total}")
    if total > 30000:
     break

if total > 30000:
        print(f"target reached stopped at {total}")
elif total==30000:
          print(f"exact target reachedstopped at {total}")
else: 
     print("target not reached")  
>>>>>>> 49053e765507a6eab96639d66c089aad6e11883c
