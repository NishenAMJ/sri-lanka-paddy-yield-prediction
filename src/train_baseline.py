
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

DATA_DIR = Path("data/processed")

# Load test dataset
test_df = pd.read_csv(
    DATA_DIR / "paddy_test.csv"
)

# Actual yields
y_test = test_df["average_yield_kg_ha"]

# Baseline: predict the previous observed yield
y_pred = test_df["previous_yield"]

# Evaluate predictions
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("===== BASELINE MODEL RESULTS =====")
print(f"MAE: {mae:.2f} kg/ha")
print(f"RMSE: {rmse:.2f} kg/ha")
print(f"R² Score: {r2:.4f}")

# Save predictions for comparison
results = test_df[
    ["district", "year", "season"]
].copy()

results["actual_yield"] = y_test
results["predicted_yield"] = y_pred

results.to_csv(
    DATA_DIR / "baseline_predictions.csv",
    index=False
)

print("\nSaved baseline predictions.")
