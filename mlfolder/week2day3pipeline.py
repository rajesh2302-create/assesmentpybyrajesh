import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Sample raw training data (contains missing values & mixed types)
X = pd.DataFrame({
    'age': [25, np.nan, 45, 35, 50, 23, 40, np.nan, 30, 55],
    'city': ['NY', 'LA', 'NY', np.nan, 'SF', 'LA', 'SF', 'NY', 'LA', 'SF'],
})
y = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]

# 2. Build ColumnTransformer & full Pipeline
num_pipe = Pipeline(
    [('imp', SimpleImputer()), ('scale', StandardScaler())]
)
cat_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(
    [('num', num_pipe, ['age']), ('cat', cat_pipe, ['city'])]
)
model = Pipeline([('prep', preprocessor), ('clf', LogisticRegression())])

# 3. 5-fold CV (fits preprocessing strictly inside each fold -> zero leakage)
scores = cross_val_score(model, X, y, cv=5)
print(f'Pipeline CV accuracy: {scores.mean():.2f} +/- {scores.std():.2f}')

# 4. Fit, save, reload, and predict on 3 raw rows (including unseen category 'TX')
model.fit(X, y)
joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')
print(loaded_model)
print('Learned weights:', loaded_model.named_steps['clf'].coef_)

new_rows = pd.DataFrame({'age': [22, 48, np.nan], 'city': ['LA', 'SF', 'TX']})
print('Predictions for new rows:', loaded_model.predict(new_rows))
