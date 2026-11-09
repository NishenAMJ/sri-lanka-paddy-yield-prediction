
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load saved Linear Regression V2 predictions
df = pd.read_csv(
    "data/processed/linear_v2_errors.csv"
)

# Separate 2022 from other years
crisis_df = df[df["season_year"] == 2022]

normal_df = df[df["season_year"] != 2022]


def evaluate(data, label):
    actual = data["actual_yield"]
    predicted = data["predicted_yield"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n===== {label} =====")
    print(f"Observations: {len(data)}")
    print(f"MAE: {mae:.2f} kg/ha")
    print(f"RMSE: {rmse:.2f} kg/ha")
    print(f"R²: {r2:.4f}")


evaluate(df, "ALL TEST YEARS (2020-2026)")
evaluate(crisis_df, "2022 ONLY")
evaluate(normal_df, "EXCLUDING 2022")

# Share of total absolute prediction error from 2022
total_error = df["absolute_error"].sum()
crisis_error = crisis_df["absolute_error"].sum()

print("\n===== 2022 ERROR CONTRIBUTION =====")
print(
    f"2022 contributes "
    f"{crisis_error / total_error * 100:.2f}% "
    f"of total absolute error"
)
