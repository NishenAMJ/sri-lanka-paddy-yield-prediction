
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD DATASETS
# ==========================================

train = pd.concat([
    pd.read_csv("data/processed/paddy_train_v2.csv"),
    pd.read_csv("data/processed/paddy_validation_v2.csv")
], ignore_index=True)

test = pd.read_csv("data/processed/paddy_test_v2.csv")

# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

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

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# ==========================================
# 3. PREPROCESSING FUNCTION
# ==========================================

def make_preprocessor(scale=False):

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        )
    ]

    if scale:
        steps.append(
            ("scaler", StandardScaler())
        )

    return ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        (
            "numerical",
            Pipeline(steps),
            numerical
        )
    ])

# ==========================================
# 4. DEFINE MODELS
# ==========================================

models = {
    "Linear Regression V2": Pipeline([
        (
            "preprocessing",
            make_preprocessor(scale=True)
        ),
        (
            "regression",
            LinearRegression()
        )
    ]),

    "Random Forest V2": Pipeline([
        (
            "preprocessing",
            make_preprocessor(scale=False)
        ),
        (
            "regression",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        )
    ])
}

# ==========================================
# 5. TRAIN AND EVALUATE MODELS
# ==========================================

print("===== V2 TEST COMPARISON =====")

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"MAE: {mae:.2f} kg/ha")
    print(f"RMSE: {rmse:.2f} kg/ha")
    print(f"R² Score: {r2:.4f}")

# ==========================================
# 6. LINEAR REGRESSION ERROR ANALYSIS
# ==========================================

# Model is already trained in the loop above
# Analyze Linear Regression V2 errors by year
linear_model = models["Linear Regression V2"]

# Train the model before predicting
linear_model.fit(X_train, y_train)

# Generate predictions
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

# ==========================================
# 7. ERROR BY YEAR
# ==========================================

print("\n===== V2 ERROR BY YEAR =====")

print(
    results.groupby("season_year")["absolute_error"]
    .agg(["mean", "count"])
    .round(2)
)

# ==========================================
# 8. ERROR BY SEASON
# ==========================================

print("\n===== V2 ERROR BY SEASON =====")

print(
    results.groupby("season")["absolute_error"]
    .mean()
    .round(2)
)

# ==========================================
# 9. ERROR BY DISTRICT
# ==========================================

print("\n===== TOP 10 DISTRICTS BY ERROR =====")

print(
    results.groupby("district")["absolute_error"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .round(2)
)

# ==========================================
# 10. LARGEST INDIVIDUAL ERRORS
# ==========================================

print("\n===== TOP 10 LARGEST V2 ERRORS =====")

print(
    results.nlargest(10, "absolute_error")[
        [
            "district",
            "year",
            "season",
            "actual_yield",
            "predicted_yield",
            "absolute_error"
        ]
    ].round(2).to_string(index=False)
)

# ==========================================
# 11. SAVE ERROR ANALYSIS
# ==========================================

results.to_csv(
    "data/processed/linear_v2_errors.csv",
    index=False
)

print("\nError analysis saved successfully.")
