import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Load same data and split (80% train, 20% test)
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# 2. Train Single Decision Tree vs Random Forest (n_estimators=200, oob_score=True)
dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42).fit(X_train, y_train)

print("--- Model Comparison ---")
print(f"Single Decision Tree Test Accuracy : {dt.score(X_test, y_test):.4f}")
print(f"Random Forest Test Accuracy        : {rf.score(X_test, y_test):.4f}")
print(f"Random Forest OOB Score            : {rf.oob_score_:.4f}")

# 3. Try n_estimators = [10, 50, 100, 300] and plot accuracy
n_list = [10, 50, 100, 300]
accuracies = []

for n in n_list:
    model = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train, y_train)
    accuracies.append(model.score(X_test, y_test))

plt.figure(figsize=(12, 5))

# Subplot 1: Accuracy vs n_estimators
plt.subplot(1, 2, 1)
plt.plot(n_list, accuracies, marker='o', color='green', linewidth=2)
plt.title("Test Accuracy vs Number of Trees")
plt.xlabel("n_estimators")
plt.ylabel("Test Accuracy")
plt.grid(True)

# 4. Plot Feature Importances as a Bar Chart
importances = pd.Series(rf.feature_importances_, index=wine.feature_names).sort_values()

plt.subplot(1, 2, 2)
importances.plot(kind='barh', color='skyblue')
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance Score")

plt.tight_layout()
plt.savefig("random_forest_plots.png", dpi=300)
plt.show()