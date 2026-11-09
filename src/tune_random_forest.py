
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    PredefinedSplit
)
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

# Combine train and validation for predefined split
combined = pd.concat([train_df, val_df], ignore_index=True)

X = combined[features]
y = combined[target]

# Training rows = -1, validation rows = 0
test_fold = np.concatenate([
    np.full(len(train_df), -1),
    np.zeros(len(val_df), dtype=int)
])

cv = PredefinedSplit(test_fold)

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

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    (
        "regression",
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
    )
])

param_distributions = {
    "regression__n_estimators": [100, 200, 300],
    "regression__max_depth": [5, 10, 15, None],
    "regression__min_samples_split": [2, 5, 10],
    "regression__min_samples_leaf": [1, 2, 4]
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="neg_mean_absolute_error",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    refit=False
)

search.fit(X, y)

print("===== RANDOM FOREST TUNING =====")
print("Best parameters:", search.best_params_)
print(f"Validation MAE: {-search.best_score_:.2f} kg/ha")

# Fit best parameters using training data only
best_model = pipeline.set_params(**search.best_params_)

best_model.fit(train_df[features], train_df[target])

predictions = best_model.predict(val_df[features])

print("\n===== VALIDATION RESULTS =====")
print(
    "MAE:",
    round(mean_absolute_error(val_df[target], predictions), 2)
)
print(
    "RMSE:",
    round(np.sqrt(mean_squared_error(val_df[target], predictions)), 2)
)
print(
    "R²:",
    round(r2_score(val_df[target], predictions), 4)
)
