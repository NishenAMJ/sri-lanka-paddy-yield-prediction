
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")


def load_data(filename):
    return pd.read_csv(
        DATA_DIR / filename,
        dtype={"year": str}
    )


def standardize_district(df):
    df = df.copy()

    df["district"] = (
        df["district"]
        .str.strip()
        .str.upper()
    )

    replacements = {
        "KILLINOCHCHI": "KILINOCHCHI",
        "KILINOCHCHIYA": "KILINOCHCHI",
        "NUWARAELIYA": "NUWARA ELIYA",
        "MAHAWELI'H'": "MAHAWELI 'H'",
        "MAHAWELI  'H'": "MAHAWELI 'H'"
    }

    df["district"] = df["district"].replace(replacements)

    return df


yield_df = load_data("paddy_yield_clean.csv")
production_df = load_data("paddy_production_combined.csv")
harvested_df = load_data("paddy_harvested_combined.csv")
sown_df = load_data("paddy_sown_combined.csv")

yield_df = standardize_district(yield_df)
production_df = standardize_district(production_df)
harvested_df = standardize_district(harvested_df)
sown_df = standardize_district(sown_df)

# Keep district-level total yield records
yield_df = yield_df[
    (yield_df["area_type"] == "District") &
    (yield_df["irrigation"] == "Total") &
    (yield_df["yield_valid_for_model"] == True)
].copy()

# Keep Total irrigation records for extent data
harvested_df = harvested_df[
    harvested_df["irrigation"] == "Total"
].copy()

sown_df = sown_df[
    sown_df["irrigation"] == "Total"
].copy()

keys = ["district", "year", "season"]

# Check uniqueness before merging
for name, df in [
    ("Yield", yield_df),
    ("Production", production_df),
    ("Harvested", harvested_df),
    ("Sown", sown_df)
]:
    duplicates = df.duplicated(subset=keys).sum()
    print(f"{name} duplicate keys: {duplicates}")

    if duplicates:
        raise ValueError(
            f"{name} contains {duplicates} duplicate keys"
        )

# Merge while preserving all valid yield records
merged_df = yield_df.merge(
    production_df[keys + ["production_1000_mt"]],
    on=keys,
    how="left",
    validate="one_to_one"
)

merged_df = merged_df.merge(
    harvested_df[keys + ["harvested_extent_ha"]],
    on=keys,
    how="left",
    validate="one_to_one"
)

merged_df = merged_df.merge(
    sown_df[keys + ["sown_extent_ha"]],
    on=keys,
    how="left",
    validate="one_to_one"
)

columns = [
    "district",
    "year",
    "season",
    "average_yield_kg_ha",
    "production_1000_mt",
    "harvested_extent_ha",
    "sown_extent_ha"
]

merged_df = merged_df[columns]

output_path = DATA_DIR / "paddy_features_combined.csv"
merged_df.to_csv(output_path, index=False)

print("\n===== MERGE RESULTS =====")
print("Total records:", len(merged_df))
print("Unique districts:", merged_df["district"].nunique())

print("\nMissing values:")
print(merged_df.isna().sum())

print("\nRecords by season:")
print(merged_df["season"].value_counts())

print("\nSample records:")
print(merged_df.head(10).to_string(index=False))

print("\nSaved to:", output_path)
