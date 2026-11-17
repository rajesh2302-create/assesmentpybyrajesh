import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Generate a dataset with 2,500 samples (approx. 1,250 per class)
X, y = make_classification(
    n_samples=2500, 
    n_features=20, 
    n_informative=15, 
    n_redundant=5, 
    random_state=42
)

# Split into a training pool (2,000 samples) and a fixed test set (500 samples)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define the required sample sizes
sample_sizes = [50, 100, 500, len(X_train_full)]
accuracies = []

# 3. Train the classifier on each sample size and record test accuracy
for size in sample_sizes:
    X_sub = X_train_full[:size]
    y_sub = y_train_full[:size]
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_sub, y_sub)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Samples: {size:4d} | Test Accuracy: {acc:.4f}")

# 4. Plot accuracy against sample size
plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, accuracies, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

# Annotate each point with its accuracy score
for x_val, y_val in zip(sample_sizes, accuracies):
    plt.text(x_val, y_val + 0.015, f"{y_val*100:.1f}%", ha='center', fontweight='bold')

plt.title("Learning Curve: Classifier Accuracy vs. Training Sample Size", fontsize=13)
plt.xlabel("Number of Training Samples", fontsize=11)
plt.ylabel("Test Accuracy", fontsize=11)
plt.xticks(sample_sizes, labels=["50", "100", "500", f"All ({len(X_train_full)})"])
plt.ylim(0.5, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save and display the plot
plt.savefig("accuracy_vs_samples.png", dpi=300)
plt.show()