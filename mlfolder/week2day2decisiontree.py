import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Load data and split (80% train, 20% test)
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# 2. Train Gini vs Entropy and compare test accuracy
print("--- Criterion Comparison ---")
for crit in ['gini', 'entropy']:
    clf = DecisionTreeClassifier(criterion=crit, random_state=42).fit(X_train, y_train)
    print(f"Criterion: {crit:7} | Test Accuracy: {clf.score(X_test, y_test):.4f}")

# 3. Tune max_depth and min_samples_leaf
params = {'max_depth': [2, 3, 4, 5, None],  'min_samples_leaf':list (range(1,5))}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5).fit(X_train, y_train)

print("\n--- Hyperparameter Tuning ---")
print("Best Parameters:", grid.best_params_)
print(f"Best Tuned Test Accuracy: {grid.best_estimator_.score(X_test, y_test):.4f}")

# 4. Train tree with max_depth=3, plot, and save image
tree_3 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)

plt.figure(figsize=(14, 8))
plot_tree(tree_3, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)
plt.savefig("wine_tree.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. Print top 5 most important features table
importances = pd.DataFrame({
    'Feature': wine.feature_names,
    'Importance': tree_3.feature_importances_
}).sort_values(by='Importance', ascending=False).head(5)

print("\n--- Top 5 Most Important Features ---")
print(importances.to_string(index=False))
