
import pandas as pd
from pathlib import Path
import re

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def reshape_production(filename, season, district_col):
    # Read the Excel file without assigning column headers
    df = pd.read_excel(RAW_DIR / filename, header=None)

    # Years are stored in row index 2
    years = df.iloc[2].copy()

    if season == "Yala":
        if str(years.iloc[2]).strip() == "198":
            years.iloc[2] = 1980

    # Store reshaped observations
    records = []

    # Start after the year header
    for row_index in range(3, len(df)):
        district = df.iloc[row_index, district_col]

        if pd.isna(district):
            continue

        district = str(district).strip().upper()

        if district in ("", "DISTRICT"):
            continue

        # Each remaining column represents a year
        for col in range(district_col + 1, len(df.columns)):
            year = str(years.iloc[col]).strip()

            # Handle numeric year headers such as 2003.0
            if re.fullmatch(r"\d{4}\.0", year):
                year = year[:-2]

            # Accept YYYY or YYYY/YYYY
            if not re.fullmatch(r"\d{4}(?:/\d{4})?", year):
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
                "production_1000_mt": value
            })

    return pd.DataFrame(records)


# Maha: district column is 1
maha = reshape_production(
    "Production-Maha_Season-1979-2025.xlsx",
    "Maha",
    1
)

# Yala: district column is 0
yala = reshape_production(
    "Production-Yala_Season-1979-2025.xlsx",
    "Yala",
    0
)

production_df = pd.concat(
    [maha, yala],
    ignore_index=True
)

production_df = production_df.sort_values(
    ["district", "season", "year"]
).reset_index(drop=True)

production_df.to_csv(
    OUTPUT_DIR / "paddy_production_combined.csv",
    index=False
)

print(production_df.head(10))
print("\nTotal records:", len(production_df))
print("\nRecords by season:")
print(production_df["season"].value_counts())
print("\nDuplicate records:")
print(production_df.duplicated(
    subset=["district", "year", "season"]
).sum())

print("\n=== YALA YEAR CHECK ===")

yala_years = sorted(
    production_df.loc[
        production_df["season"] == "Yala", "year"
    ].unique()
)

print("1980 present:", "1980" in yala_years)
print("1979 present:", "1979" in yala_years)
print("1981 present:", "1981" in yala_years)

print("\n=== YALA 1980 RECORDS ===")

print(
    production_df[
        (production_df["season"] == "Yala") &
        (production_df["year"] == "1980")
    ].head(10).to_string(index=False)
)