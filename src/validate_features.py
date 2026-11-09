
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

df = pd.read_csv(
    DATA_DIR / "paddy_features_combined.csv",
    dtype={"year": str}
)

print("===== DATASET SUMMARY =====")
print("Shape:", df.shape)
print("\nData types:")
print(df.dtypes)

print("\n===== MISSING PRODUCTION =====")
missing = df[df["production_1000_mt"].isna()]

print("Total missing:", len(missing))
print("\nMissing by district:")
print(missing["district"].value_counts())

print("\nMissing by season:")
print(missing["season"].value_counts())

print("\n===== EXTENT VALIDATION =====")
invalid_extent = df[
    df["harvested_extent_ha"] > df["sown_extent_ha"]
]

print("Harvested > Sown:", len(invalid_extent))
print(invalid_extent[
    ["district", "year", "season",
     "harvested_extent_ha", "sown_extent_ha"]
].head(15).to_string(index=False))

print("\n===== YIELD SUMMARY =====")
print(df["average_yield_kg_ha"].describe())

print("\n===== PRODUCTION CONSISTENCY =====")

# Convert reported production from thousand metric tonnes to kg.
# 1 thousand metric tonnes = 1,000,000 kg.
df["implied_yield_kg_ha"] = (
    df["production_1000_mt"] * 1_000_000
    / df["harvested_extent_ha"].replace(0, float("nan"))
)

df["yield_difference_pct"] = (
    (df["implied_yield_kg_ha"] - df["average_yield_kg_ha"])
    / df["average_yield_kg_ha"] * 100
)

large_differences = df[
    df["yield_difference_pct"].abs() > 20
]

print("Records with >20% difference:", len(large_differences))

print(large_differences[
    ["district", "year", "season",
     "average_yield_kg_ha",
     "implied_yield_kg_ha",
     "yield_difference_pct"]
].head(15).to_string(index=False))
