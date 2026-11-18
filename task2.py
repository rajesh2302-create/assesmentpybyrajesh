import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 0. Dataset with Missing Values, Dates, Ordered & Nominal Categories
# ---------------------------------------------------------
def get_raw_loan_data(n=1500, random_state=42):
    rng = np.random.RandomState(random_state)
    income = rng.lognormal(mean=10.8, sigma=0.6, size=n)
    loan_amount = rng.lognormal(mean=9.5, sigma=0.7, size=n)
    credit_score = rng.normal(loc=650, scale=75, size=n).clip(300, 850)
    
    education = rng.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n)
    home_ownership = rng.choice(['RENT', 'MORTGAGE', 'OWN'], size=n)
    
    # Generate application dates
    dates = pd.to_datetime('2024-01-01') + pd.to_timedelta(rng.randint(0, 700, size=n), unit='D')
    
    # Target
    dti = loan_amount / (income + 1)
    prob = 1 / (1 + np.exp(-(-1.8 + 2.5 * dti - 0.008 * (credit_score - 600))))
    default = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'application_date': dates.astype(str),
        'income': income,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'education': education,
        'home_ownership': home_ownership,
        'default': default
    })
    
    # Inject ~8% missing values to test imputers
    for col in ['income', 'loan_amount', 'credit_score', 'education', 'home_ownership']:
        mask = rng.rand(n) < 0.08
        df.loc[mask, col] = np.nan
        
    return df

df = get_raw_loan_data()
X = df.drop(columns=['default'])
y = df['default']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# ---------------------------------------------------------
# 1. Custom Feature Engineering Transformer (Runs inside Pipeline)
# ---------------------------------------------------------
class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Feature 1: Date decomposition (Year, Month, Day of Week)
        dt = pd.to_datetime(X['application_date'], errors='coerce')
        X['app_month'] = dt.dt.month.fillna(1)
        X['app_dayofweek'] = dt.dt.dayofweek.fillna(0)
        
        # Feature 2: Interaction feature (income / loan_amount)
        X['income_to_loan_ratio'] = X['income'] / (X['loan_amount'] + 1e-5)
        
        # Feature 3: Log transform for skewed column (income)
        X['log_income'] = np.log1p(X['income'].clip(lower=0))
        
        return X.drop(columns=['application_date'])

# ---------------------------------------------------------
# 2. Define ColumnTransformer (Imputation, Encoding, Scaling)
# ---------------------------------------------------------
numeric_features = ['income', 'loan_amount', 'credit_score', 'app_month', 'app_dayofweek', 'income_to_loan_ratio', 'log_income']
ordinal_features = ['education']
nominal_features = ['home_ownership']

# Numeric pipeline: SimpleImputer (median) + StandardScaler
num_pipeline_simple = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Numeric pipeline alternative: KNNImputer + StandardScaler
num_pipeline_knn = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# Ordinal pipeline: Impute most_frequent + OrdinalEncoder with explicit order
ord_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[['High School', 'Bachelor', 'Master', 'PhD']], handle_unknown='use_encoded_value', unknown_value=-1))
])

# Nominal pipeline: Impute most_frequent + OneHotEncoder
nom_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Build full Pipeline with SimpleImputer
preprocessor_simple = ColumnTransformer([
    ('num', num_pipeline_simple, numeric_features),
    ('ord', ord_pipeline, ordinal_features),
    ('nom', nom_pipeline, nominal_features)
])

full_pipeline_simple = Pipeline([
    ('feat_eng', LoanFeatureEngineer()),
    ('prep', preprocessor_simple),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Build full Pipeline with KNNImputer for comparison
preprocessor_knn = ColumnTransformer([
    ('num', num_pipeline_knn, numeric_features),
    ('ord', ord_pipeline, ordinal_features),
    ('nom', nom_pipeline, nominal_features)
])

full_pipeline_knn = Pipeline([
    ('feat_eng', LoanFeatureEngineer()),
    ('prep', preprocessor_knn),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# ---------------------------------------------------------
# 3. Cross-Validation Inside Pipeline (Zero Leakage)
# ---------------------------------------------------------
cv_simple = cross_val_score(full_pipeline_simple, X_train, y_train, cv=5, scoring='accuracy')
cv_knn = cross_val_score(full_pipeline_knn, X_train, y_train, cv=5, scoring='accuracy')

print(f"Pipeline CV accuracy (SimpleImputer median): {cv_simple.mean():.2f}")
print(f"Pipeline CV accuracy (KNNImputer k=5):       {cv_knn.mean():.2f}")

# ---------------------------------------------------------
# 4. Fit Final Pipeline, Save with Joblib & Predict 3 Raw Rows
# ---------------------------------------------------------
full_pipeline_knn.fit(X_train, y_train)
model_filename = "loan_default_pipeline.joblib"
joblib.dump(full_pipeline_knn, model_filename)
print(f"\nSaved fitted pipeline to '{model_filename}'")

# Load pipeline and predict on 3 brand-new raw rows (including missing values)
loaded_pipeline = joblib.load(model_filename)

new_raw_rows = pd.DataFrame([
    {'application_date': '2025-03-15', 'income': 95000.0, 'loan_amount': 12000.0, 'credit_score': 740.0, 'education': 'Master', 'home_ownership': 'OWN'},
    {'application_date': '2025-06-22', 'income': 22000.0, 'loan_amount': 35000.0, 'credit_score': 510.0, 'education': 'High School', 'home_ownership': 'RENT'},
    {'application_date': '2025-09-10', 'income': np.nan,  'loan_amount': 15000.0, 'credit_score': 660.0, 'education': 'Bachelor', 'home_ownership': np.nan}
])

predictions = loaded_pipeline.predict(new_raw_rows)
print(f"Predictions for 3 new rows: {predictions}")
