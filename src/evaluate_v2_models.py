
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use all pre-2020 data for training
train = pd.concat([
    pd.read_csv("data/processed/paddy_train_v2.csv"),
    pd.read_csv("data/processed/paddy_validation_v2.csv")
], ignore_index=True)

test = pd.read_csv("data/processed/paddy_test_v2.csv")

features = [
    "district", "season", "season_year",
    "sown_extent_ha", "season_gap",
    "previous_yield", "yield_3season_avg"
]

target = "average_yield_kg_ha"

categorical = ["district", "season"]
numerical = [
    "season_year", "sown_extent_ha",
    "season_gap", "previous_yield",
    "yield_3season_avg"
]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

def make_preprocessor(scale=False):
    steps = [
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        ))
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical),
        ("numerical", Pipeline(steps), numerical)
    ])

models = {
    "Linear Regression V2": Pipeline([
        ("preprocessing", make_preprocessor(scale=True)),
        ("regression", LinearRegression())
    ]),
    "Random Forest V2": Pipeline([
        ("preprocessing", make_preprocessor(scale=False)),
        ("regression", RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# Analyze Linear Regression V2 errors by year
linear_model = models["Linear Regression V2"]

# Train Linear Regression first
linear_model.fit(X_train, y_train)

# Predict after training
predictions = linear_model.predict(X_test)

results = test[
    ["district", "year", "season", "season_year"]
].copy()

results["actual_yield"] = y_test.to_numpy()
results["predicted_yield"] = predictions

results["absolute_error"] = (
    results["actual_yield"] -
    results["predicted_yield"]
).abs()

print("\n===== V2 ERROR BY YEAR =====")
print(
    results.groupby("season_year")["absolute_error"]
    .agg(["mean", "count"])
    .round(2)
)

print("\n===== V2 ERROR BY SEASON =====")
print(
    results.groupby("season")["absolute_error"]
    .mean()
    .round(2)
)

print("\n===== TOP 10 LARGEST V2 ERRORS =====")
print(
    results.nlargest(10, "absolute_error")[
        ["district", "year", "season",
         "actual_yield", "predicted_yield",
         "absolute_error"]
    ].round(2).to_string(index=False)
)

results.to_csv(
    "data/processed/linear_v2_errors.csv",
    index=False
)

print("===== V2 TEST COMPARISON =====")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"\n{name}")
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f} kg/ha")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f} kg/ha")
    print(f"R²: {r2_score(y_test, pred):.4f}")
