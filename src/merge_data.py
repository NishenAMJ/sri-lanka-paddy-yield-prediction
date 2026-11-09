
import pandas as pd
from pathlib import Path
import re

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def reshape_paddy_file(filename, season):
    path = RAW_DIR / filename

    # Read Excel without assuming header positions
    df = pd.read_excel(path, header=None)

    # Your logs show year headers at row index 2
    # and irrigation categories at row index 3.
    years = df.iloc[2].ffill().copy()

    if season == "Maha":
        years = years.replace({
            "2000/2021": "2020/2021"
        })
    categories = df.iloc[3]

    records = []

    for row_index in range(4, len(df)):
        district = df.iloc[row_index, 1]

        if pd.isna(district):
            continue

        district = str(district).strip().upper()

        # Skip aggregate and explanatory rows
        if district in ("DISTRICT", ""):
            continue

        for col in range(2, len(df.columns)):
            year = str(years.iloc[col]).strip()
            category = str(categories.iloc[col]).strip().title()

            if not re.fullmatch(r"\d{4}(?:/\d{4})?", year):
                continue

            if category not in (
                "Major", "Minor", "Rainfed", "Total"
            ):
                continue

            value = pd.to_numeric(
                df.iloc[row_index, col],
                errors="coerce"
            )

            if pd.isna(value):
                continue

            records.append({
                "district": district,
                "year": year,
                "season": season,
                "irrigation": category,
                "average_yield_kg_ha": value
            })

    return pd.DataFrame(records)



# Process Maha yield
maha = reshape_paddy_file(
    "Average_Yeild-Maha_Season-1979-2025.xlsx",
    "Maha"
)

# Process Yala yield
yala = reshape_paddy_file(
    "Average_Yeild-Yala_Season-1979-2025.xlsx",
    "Yala"
)

# Combine both datasets vertically
yield_df = pd.concat(
    [maha, yala],
    ignore_index=True
)

# Sort for easier inspection
yield_df = yield_df.sort_values(
    by=["district", "season", "year", "irrigation"]
).reset_index(drop=True)

print(yield_df.head(15))
print("\nTotal rows:", len(yield_df))
print("\nRows by season:")
print(yield_df["season"].value_counts())

# Save the combined dataset
yield_df.to_csv(
    OUTPUT_DIR / "paddy_yield_combined.csv",
    index=False
)


print("\nMissing values:")
print(yield_df.isnull().sum())

print("\nDuplicate records:")
print(
    yield_df.duplicated(
        subset=["district", "year", "season", "irrigation"]
    ).sum()
)

print("\nUnique districts:")
print(sorted(yield_df["district"].unique()))

print("\nIrrigation categories:")
print(yield_df["irrigation"].value_counts())

print("\nYield summary:")
print(yield_df["average_yield_kg_ha"].describe())
