
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

train_df = pd.read_csv("data/processed/paddy_train.csv")
test_df = pd.read_csv("data/processed/paddy_test.csv")

features = [
    "district",
    "season",
    "season_year",
    "sown_extent_ha",
    "previous_yield",
    "yield_3season_avg"
]

target = "average_yield_kg_ha"

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["district", "season"]
        )
    ],
    remainder="passthrough"
)

# Build Random Forest pipeline
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "regression",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("===== RANDOM FOREST RESULTS =====")
print(f"MAE: {mae:.2f} kg/ha")
print(f"RMSE: {rmse:.2f} kg/ha")
print(f"R² Score: {r2:.4f}")

