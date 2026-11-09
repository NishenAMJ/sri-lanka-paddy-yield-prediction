
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")

df = pd.read_csv(
    DATA_DIR / "paddy_features_combined.csv",
    dtype={"year": str}
)

# 1. Find missing production records
missing = df[df["production_1000_mt"].isna()]

print("===== MISSING PRODUCTION YEARS =====")
print(
    missing.groupby(["season", "year"])
    .size()
    .sort_values(ascending=False)
    .to_string()
)

print("\n===== MISSING PRODUCTION RECORDS =====")
print(
    missing[["district", "year", "season"]]
    .to_string(index=False)
)

# 2. Compare reported production with expected production
# kg/ha * hectares / 1,000,000 = thousand metric tonnes
df["expected_production_1000_mt"] = (
    df["average_yield_kg_ha"]
    * df["harvested_extent_ha"]
    / 1_000_000
)

valid = df[
    (df["production_1000_mt"] > 0) &
    (df["expected_production_1000_mt"] > 0)
].copy()

valid["production_ratio"] = (
    valid["production_1000_mt"]
    / valid["expected_production_1000_mt"]
)

valid["difference_pct"] = (
    (valid["production_ratio"] - 1) * 100
)

valid["year_start"] = (
    valid["year"].str[:4].astype(int)
)

valid["decade"] = (
    valid["year_start"] // 10 * 10
)

print("\n===== PRODUCTION RATIO SUMMARY =====")
print(valid["production_ratio"].describe())

print("\n===== MEDIAN RATIO BY DECADE =====")
print(
    valid.groupby(["season", "decade"])["production_ratio"]
    .median()
    .to_string()
)

print("\n===== MEDIAN RATIO BY DISTRICT =====")
print(
    valid.groupby("district")["production_ratio"]
    .median()
    .sort_values()
    .to_string()
)

print("\n===== LARGE DIFFERENCES BY SEASON =====")
print(
    valid.assign(
        large_difference=valid["difference_pct"].abs() > 20
    )
    .groupby("season")["large_difference"]
    .agg(["sum", "count", "mean"])
    .to_string()
)

print("\n===== RECENT AMPARA RECORDS =====")
print(
    valid[
        (valid["district"] == "AMPARA") &
        (valid["year_start"] >= 2018)
    ][
        [
            "year",
            "season",
            "production_1000_mt",
            "expected_production_1000_mt",
            "production_ratio"
        ]
    ].to_string(index=False)
)
