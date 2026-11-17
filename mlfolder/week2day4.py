from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split

# 1. Create imbalanced dataset (95% Class 0, 5% Class 1)
X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 2. Majority-class baseline model
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
print(f"Majority-class model: accuracy {accuracy_score(y_test, y_dummy):.2f}, "
      f"precision {precision_score(y_test, y_dummy, zero_division=0):.2f}, "
      f"recall {recall_score(y_test, y_dummy):.2f}, "
      f"F1 {f1_score(y_test, y_dummy):.2f}\n")

# 3. Standard LogisticRegression
lr = LogisticRegression().fit(X_train, y_train)
y_lr = lr.predict(X_test)
print("--- Standard LogisticRegression ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_lr))
print(classification_report(y_test, y_lr, zero_division=0))

# 4. Balanced LogisticRegression
lr_bal = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
y_bal = lr_bal.predict(X_test)
print("--- Balanced LogisticRegression ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_bal))
print(classification_report(y_test, y_bal))
