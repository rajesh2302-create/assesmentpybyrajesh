import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Dataset-ai load panni 80/20 ratio-la split panrom
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Baseline Model (Always predicts the mean of y_train)
baseline_model = DummyRegressor(strategy="mean")
baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_test)

# 3. Linear Regression Model train panrom
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 4. Metrics calculate panrom
results = [
    {
        "Model": "Baseline (Mean)",
        "MAE": round(mean_absolute_error(y_test, y_pred_base), 2),
        "RMSE": round(root_mean_squared_error(y_test, y_pred_base), 2),
        "R2": round(r2_score(y_test, y_pred_base), 2),
    },
    {
        "Model": "Linear Regression",
        "MAE": round(mean_absolute_error(y_test, y_pred_lr), 2),
        "RMSE": round(root_mean_squared_error(y_test, y_pred_lr), 2),
        "R2": round(r2_score(y_test, y_pred_lr), 2),
    },
]

# 5. Output Table display panrom
df_metrics = pd.DataFrame(results)
print(df_metrics.to_string(index=False))
