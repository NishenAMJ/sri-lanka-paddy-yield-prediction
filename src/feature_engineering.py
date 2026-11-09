
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

df = pd.read_csv(
    DATA_DIR / "paddy_eda.csv",
    dtype={"year": str}
)

# Ensure correct chronological order
# Use the ending year for Maha seasons
# and the calendar year for Yala seasons
df["season_year"] = df["year"].apply(
    lambda x: int(x.split("/")[-1])
)

# Maha occurs before Yala in the same ending year
season_order = {"Maha": 0, "Yala": 1}
df["season_order"] = df["season"].map(season_order)

df = df.sort_values(
    ["district", "season_year", "season_order"]
).reset_index(drop=True)

# Historical yield for each district
grouped = df.groupby("district")["average_yield_kg_ha"]

# Previous observed season's yield
df["previous_yield"] = grouped.shift(1)

# Average of up to 3 previous observations
df["yield_3season_avg"] = (
    df.groupby("district")["average_yield_kg_ha"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

# Remove temporary sorting column
df = df.drop(columns=["season_order"])

print("===== FEATURE ENGINEERING =====")
print("Shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nSample records:")
print(
    df[
        [
            "district",
            "year",
            "season",
            "average_yield_kg_ha",
            "previous_yield",
            "yield_3season_avg"
        ]
    ].head(10).to_string(index=False)
)

output_path = DATA_DIR / "paddy_ml_features.csv"
df.to_csv(output_path, index=False)

print("\nSaved to:", output_path)
