
import pandas as pd
from pathlib import Path
import re

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def reshape_harvested(filename, season):

    df = pd.read_excel(RAW_DIR / filename, header=None)

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

        if district in ("", "DISTRICT"):
            continue

        for col in range(2, len(df.columns)):

            year = str(years.iloc[col]).strip()
            category = str(categories.iloc[col]).strip().title()

            if not re.fullmatch(r"\d{4}(?:/\d{4})?", year):
                continue

            if category not in ("Major", "Minor", "Rainfed", "Total"):
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
                "harvested_extent_ha": value
            })

    return pd.DataFrame(records)

maha = reshape_harvested(
    "Harvested_Extent-Maha_Seasons-1979-2025.xlsx",
    "Maha"
)

yala = reshape_harvested(
    "Harvested_Extent-Yala_Seasons-1979-2025.xlsx",
    "Yala"
)

harvested_df = pd.concat(
    [maha, yala],
    ignore_index=True
)

harvested_df = harvested_df.sort_values(
    ["district", "season", "year", "irrigation"]
).reset_index(drop=True)

harvested_df.to_csv(
    OUTPUT_DIR / "paddy_harvested_combined.csv",
    index=False
)

print(harvested_df.head(10))

print("\nTotal records:", len(harvested_df))

print("\nRecords by season:")
print(harvested_df["season"].value_counts())

print("\nDuplicate records:")
print(harvested_df.duplicated(
    subset=["district", "year", "season", "irrigation"]
).sum())

print("\nSuspicious year records:")
print((harvested_df["year"] == "2000/2021").sum())
