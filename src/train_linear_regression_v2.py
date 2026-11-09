
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load V2 datasets
train_df = pd.read_csv(
    "data/processed/paddy_train_v2.csv"
)

val_df = pd.read_csv(
    "data/processed/paddy_validation_v2.csv"
)

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

categorical_features = ["district", "season"]

numerical_features = [
    "season_year",
    "sown_extent_ha",
    "season_gap",
    "previous_yield",
    "yield_3season_avg"
]

# Imputation and scaling
numerical_pipeline = Pipeline([
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
        categorical_features
    ),
    (
        "numerical",
        numerical_pipeline,
        numerical_features
    )
])

# Linear Regression
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])

# Train on 1979–2014
model.fit(train_df[features], train_df[target])

# Validate on 2015–2019
predictions = model.predict(val_df[features])

mae = mean_absolute_error(val_df[target], predictions)
rmse = np.sqrt(
    mean_squared_error(val_df[target], predictions)
)
r2 = r2_score(val_df[target], predictions)

print("===== LINEAR REGRESSION V2 VALIDATION =====")
print(f"MAE: {mae:.2f} kg/ha")
print(f"RMSE: {rmse:.2f} kg/ha")
print(f"R² Score: {r2:.4f}")
