
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Paths
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Use all pre-2020 data
train_df = pd.concat([
    pd.read_csv(DATA_DIR / "paddy_train_v2.csv"),
    pd.read_csv(DATA_DIR / "paddy_validation_v2.csv")
], ignore_index=True)

features = [
    "district",
    "season",
    "season_year",
    "sown_extent_ha",
    "season_gap",
    "previous_yield",
    "yield_3season_avg"
]

target = "average_yield_kg_ha"

categorical = ["district", "season"]

numerical = [
    "season_year",
    "sown_extent_ha",
    "season_gap",
    "previous_yield",
    "yield_3season_avg"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="median",
        add_indicator=True
    )),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical
    ),
    (
        "numerical",
        numeric_pipeline,
        numerical
    )
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])

# Train the model
model.fit(train_df[features], train_df[target])

# Save fitted model
model_path = MODEL_DIR / "paddy_yield_linear_v2.joblib"
joblib.dump(model, model_path)

print("===== MODEL SAVED =====")
print("Model:", model_path)
print("Training records:", len(train_df))
print("Features:", features)

# Verify loading
loaded_model = joblib.load(model_path)

sample = train_df[features].iloc[[0]]
prediction = loaded_model.predict(sample)

print("\n===== MODEL LOAD TEST =====")
print(f"Sample predicted yield: {prediction[0]:.2f} kg/ha")
print("Model loaded and predicted successfully.")
