
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

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ))
])

# Retrain on all observations before 2020
model.fit(train_df[features], train_df[target])

# Evaluate on 2020–2026
predictions = model.predict(test_df[features])

mae = mean_absolute_error(test_df[target], predictions)
rmse = np.sqrt(mean_squared_error(test_df[target], predictions))
r2 = r2_score(test_df[target], predictions)

print("===== TUNED RANDOM FOREST TEST RESULTS =====")
print(f"MAE: {mae:.2f} kg/ha")
print(f"RMSE: {rmse:.2f} kg/ha")
print(f"R² Score: {r2:.4f}")
