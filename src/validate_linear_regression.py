
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

train_df = pd.read_csv("data/processed/paddy_train_inner.csv")
val_df = pd.read_csv("data/processed/paddy_validation.csv")

features = [
    "district",
    "season",
    "season_year",
    "sown_extent_ha",
    "previous_yield",
    "yield_3season_avg"
]

target = "average_yield_kg_ha"

categorical_features = ["district", "season"]

numerical_features = [
    "season_year",
    "sown_extent_ha",
    "previous_yield",
    "yield_3season_avg"
]

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "numerical",
        StandardScaler(),
        numerical_features
    )
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])

# Train only on data before 2015
model.fit(train_df[features], train_df[target])

# Predict 2015–2019
predictions = model.predict(val_df[features])

print("===== LINEAR REGRESSION VALIDATION =====")
print(
    f"MAE: {mean_absolute_error(val_df[target], predictions):.2f} kg/ha"
)
print(
    f"RMSE: {np.sqrt(mean_squared_error(val_df[target], predictions)):.2f} kg/ha"
)
print(
    f"R² Score: {r2_score(val_df[target], predictions):.4f}"
)
