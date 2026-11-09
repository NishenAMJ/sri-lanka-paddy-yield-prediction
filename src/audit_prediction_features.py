
import pandas as pd

df = pd.read_csv(
    "data/processed/paddy_ml_features_v2.csv"
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

print("===== PREDICTION FEATURE AUDIT =====")

print("\nTotal observations:", len(df))

print("\nFeature availability:")
for feature in features:
    available = df[feature].notna().sum()
    percentage = available / len(df) * 100

    print(
        f"{feature}: "
        f"{available}/{len(df)} "
        f"({percentage:.2f}%)"
    )

print("\n===== COMPLETE HISTORICAL FEATURES =====")

complete = df[
    ["previous_yield", "yield_3season_avg"]
].notna().all(axis=1)

print("Records with both historical features:", complete.sum())

print("Records missing at least one:", (~complete).sum())

print("\n===== RECENT DATA AVAILABILITY =====")

recent = df[df["season_year"] >= 2020]

print(
    recent.groupby("season_year")[
        ["previous_yield", "yield_3season_avg"]
    ].count()
)

print("\n===== DISTRICTS WITH MISSING HISTORY =====")

missing = df[
    df["previous_yield"].isna()
    | df["yield_3season_avg"].isna()
]

print(
    missing.groupby("district")
    .size()
    .sort_values(ascending=False)
    .head(10)
)
