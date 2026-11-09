
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

df = pd.read_csv(
    DATA_DIR / "paddy_features_combined.csv",
    dtype={"year": str}
)

# Exclude production from the initial EDA dataset
eda_df = df[
    [
        "district",
        "year",
        "season",
        "average_yield_kg_ha",
        "sown_extent_ha",
        "harvested_extent_ha"
    ]
].copy()

# Extract starting year for chronological analysis
eda_df["year_start"] = (
    eda_df["year"].str[:4].astype(int)
)

# Sort chronologically
eda_df = eda_df.sort_values(
    ["year_start", "season", "district"]
).reset_index(drop=True)

print("===== EDA DATASET =====")
print("Shape:", eda_df.shape)

print("\nMissing values:")
print(eda_df.isna().sum())

print("\nYield statistics:")
print(eda_df["average_yield_kg_ha"].describe())

print("\nRecords by season:")
print(eda_df["season"].value_counts())

output_path = DATA_DIR / "paddy_eda.csv"
eda_df.to_csv(output_path, index=False)

print("\nSaved to:", output_path)
