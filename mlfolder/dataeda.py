import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. Offline Telco-style Churn Dataset (No Internet Needed)
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'Contract': np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20]
    ),
    'InternetService': np.random.choice(
        ['Fiber optic', 'DSL', 'No'], n, p=[0.45, 0.35, 0.20]
    ),
    'PaymentMethod': np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n
    ),
    'tenure': np.random.randint(1, 72, n),
    'MonthlyCharges': np.round(np.random.uniform(20, 115, n), 2),
    'Churn': np.random.choice(['No', 'Yes'], n, p=[0.73, 0.27]),
})

print(
    f'Shape: {df.shape} | Nulls: {df.isnull().sum().sum()} | Dupes:'
    f' {df.duplicated().sum()}\nTypes:\n{df.dtypes.value_counts()}'
)
print(
    '\nClass Balance:\n',
    df['Churn'].value_counts(normalize=True).round(2),
)

# 2. Immediately hold out 20% test set
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['Churn'], random_state=42
)

# 3. EDA: 5 charts showing target ('Churn') vs key features
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
features = [
    'Contract',
    'InternetService',
    'PaymentMethod',
    'tenure',
    'MonthlyCharges',
]
for i, col in enumerate(features):
  if train_df[col].dtype == 'object':
    sns.countplot(data=train_df, x=col, hue='Churn', ax=axes[i])
  else:
    sns.boxplot(data=train_df, x='Churn', y=col, ax=axes[i])
  axes[i].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('eda_5_charts.png')
plt.show()
