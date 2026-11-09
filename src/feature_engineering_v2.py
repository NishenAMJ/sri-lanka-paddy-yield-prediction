
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

df = pd.read_csv(
    DATA_DIR / "paddy_eda.csv",
    dtype={"year": str}
)

# Identify the season's ending/calendar year
df["season_year"] = df["year"].apply(
    lambda x: int(x.split("/")[-1])
)

# Represent seasons in chronological order
df["season_index"] = (
    df["season_year"] * 2
    + df["season"].map({"Maha": 0, "Yala": 1})
)

df = df.sort_values(
    ["district", "season_index"]
).reset_index(drop=True)

grouped = df.groupby("district")

# Difference from the previous observed season
df["season_gap"] = grouped["season_index"].diff()

# Previous observed yield
df["previous_yield"] = grouped[
    "average_yield_kg_ha"
].shift(1)

# Only retain genuinely consecutive seasons
df.loc[
    df["season_gap"] != 1,
    "previous_yield"
] = float("nan")

# Three prior yields, initially based on observed rows
df["yield_3season_avg"] = (
    df.groupby("district")["average_yield_kg_ha"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=3
        ).mean()
    )
)

# Check all three previous gaps
gaps = df.groupby("district")["season_gap"]

valid_three_seasons = (
    df["season_gap"].eq(1)
    & gaps.shift(1).eq(1)
    & gaps.shift(2).eq(1)
)

df.loc[
    ~valid_three_seasons,
    "yield_3season_avg"
] = float("nan")

df = df.drop(columns=["season_index"])

print("===== FEATURE ENGINEERING V2 =====")
print("Shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nHistorical gap distribution:")
print(df["season_gap"].value_counts().sort_index())

output = DATA_DIR / "paddy_ml_features_v2.csv"
df.to_csv(output, index=False)

print("\nSaved to:", output)
