
import pandas as pd

df = pd.read_csv("data/processed/paddy_ml_features.csv")

# Give every cultivation season a sequential number
df["season_index"] = (
    df["season_year"] * 2
    + df["season"].map({"Maha": 0, "Yala": 1})
)

df = df.sort_values(
    ["district", "season_index"]
).copy()

# Difference between consecutive observed seasons
df["season_gap"] = (
    df.groupby("district")["season_index"].diff()
)

# Ignore first observation for each district
valid = df.dropna(subset=["season_gap"])

print("===== HISTORICAL SEASON GAPS =====")

print("\nGap distribution:")
print(valid["season_gap"].value_counts().sort_index())

print("\nObservations with gaps greater than 1:")
print((valid["season_gap"] > 1).sum())

print("\nDistricts with the most gaps:")
print(
    valid[valid["season_gap"] > 1]
    .groupby("district")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("\nLargest historical gaps:")
print(
    valid.nlargest(10, "season_gap")[
        ["district", "year", "season", "season_gap"]
    ].to_string(index=False)
)
