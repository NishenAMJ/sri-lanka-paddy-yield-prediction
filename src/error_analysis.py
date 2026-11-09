
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

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

preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     ["district", "season"]),
    ("numerical",
     StandardScaler(),
     ["season_year", "sown_extent_ha",
      "previous_yield", "yield_3season_avg"])
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])

model.fit(train_df[features], train_df[target])

results = test_df[
    ["district", "year", "season", "season_year", target]
].copy()

results["predicted_yield"] = model.predict(test_df[features])

results["absolute_error"] = (
    results[target] - results["predicted_yield"]
).abs()

print("===== ERROR BY SEASON =====")
print(
    results.groupby("season")["absolute_error"]
    .mean()
    .round(2)
)

print("\n===== TOP 10 DISTRICTS BY ERROR =====")
print(
    results.groupby("district")["absolute_error"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .round(2)
)

print("\n===== ERROR BY YEAR =====")
print(
    results.groupby("season_year")["absolute_error"]
    .mean()
    .round(2)
)

print("\n===== 10 LARGEST ERRORS =====")
print(
    results.nlargest(10, "absolute_error")[
        ["district", "year", "season",
         target, "predicted_yield", "absolute_error"]
    ].round(2).to_string(index=False)
)

results.to_csv(
    "data/processed/linear_regression_errors.csv",
    index=False
)

print("\nError analysis saved.")
