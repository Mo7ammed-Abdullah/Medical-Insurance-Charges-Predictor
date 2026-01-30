import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================
# Load dataset
# =====================
df = pd.read_csv("insurance.csv")  # change path if needed

# =====================
# Target and features
# =====================
X = df.drop("charges", axis=1)
y = df["charges"]

# =====================
# Column split
# =====================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# =====================
# Preprocessing
# =====================
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, numeric_features),
    ("cat", cat_transformer, categorical_features)
])

# =====================
# Your Gradient Boosting Model (PUT YOUR BEST PARAMS HERE)
# =====================
gb_base = GradientBoostingRegressor(
    random_state=42,
    # Replace these with search_gb.best_params_ values:
    n_estimators=200,      
    learning_rate=0.03,      
)

# Wrap to train on log1p(y) but predict in original money
gb_model = TransformedTargetRegressor(
    regressor=gb_base,
    func=np.log1p,
    inverse_func=np.expm1
)

# =====================
# Full Pipeline
# =====================
gb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", gb_model)
])

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
gb_pipeline.fit(X_train, y_train)

# =====================
# Evaluation (money scale)
# =====================
y_pred = gb_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R2  : {r2:.4f}")

# =====================
# Save model
# =====================
with open("insurance_gb_pipeline.pkl", "wb") as f:
    pickle.dump(gb_pipeline, f)

print("✅ Saved as insurance_gb_pipeline.pkl")
