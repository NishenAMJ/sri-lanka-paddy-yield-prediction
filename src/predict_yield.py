
import joblib
import pandas as pd
from pathlib import Path

# Load the saved model
MODEL_PATH = Path("models/paddy_yield_linear_v2.joblib")
model = joblib.load(MODEL_PATH)

# Example input: hypothetical Ampara Yala season
# These are illustrative values, not official statistics.
input_data = pd.DataFrame([{
    "district": "AMPARA",
    "season": "Yala",
    "season_year": 2025,
    "sown_extent_ha": 20000,
    "season_gap": 1,
    "previous_yield": 4000,
    "yield_3season_avg": 3900
}])

# Make prediction
prediction = model.predict(input_data)[0]

print("===== PADDY YIELD PREDICTION =====")
print("District:", input_data["district"][0])
print("Season:", input_data["season"][0])
print("Year:", input_data["season_year"][0])
print(f"Predicted yield: {prediction:.2f} kg/ha")
print(f"Predicted yield: {prediction / 1000:.2f} tonnes/ha")
