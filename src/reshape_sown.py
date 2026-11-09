
import pandas as pd
from pathlib import Path
import re

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_year(value):
    """Convert Excel year values into consistent strings."""

    if pd.isna(value):
        return None

    year = str(value).strip()

    # Convert numeric Excel years such as 1979.0
    if re.fullmatch(r"\d{4}\.0", year):
        year = year[:-2]

    # Remove whitespace inside year labels
    year = re.sub(r"\s+", "", year)

    # Correct the known Maha year typo
    if year == "2000/2021":
        year = "2020/2021"

    # Accept 1979 or 1978/1979
    if re.fullmatch(r"\d{4}(?:/\d{4})?", year):
        return year

    return None


def reshape_sown(filename, season):

    file_path = RAW_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_excel(file_path, header=None)

    print(f"\n===== Processing {season} =====")
    print("Excel shape:", df.shape)

    if season == "Maha":
        year_row = 1
        category_row = 2
        data_start_row = 3
    else:
        year_row = 2
        category_row = 3
        data_start_row = 4

    years = df.iloc[year_row].ffill()
    categories = df.iloc[category_row]

    print("Sample years:", years.iloc[2:10].tolist())
    print("Sample categories:", categories.iloc[2:10].tolist())

    records = []

    for row_index in range(data_start_row, len(df)):

        district = df.iloc[row_index, 1]

        if pd.isna(district):
            continue

        district = str(district).strip().upper()

        if district in ("", "DISTRICT"):
            continue

        for col in range(2, len(df.columns)):

            year = normalize_year(years.iloc[col])

            if year is None:
                continue

            category = str(categories.iloc[col]).strip().title()

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
                "sown_extent_ha": float(value)
            })

    result = pd.DataFrame(
        records,
        columns=[
            "district",
            "year",
            "season",
            "irrigation",
            "sown_extent_ha"
        ]
    )

    print(f"{season} extracted records:", len(result))

    return result


def main():

    maha = reshape_sown(
        "Sown_Extent-Maha_Seasons-1979-2025.xlsx",
        "Maha"
    )

    yala = reshape_sown(
        "Sown_Extent-Yala_Seasons-1979-2025.xlsx",
        "Yala"
    )

    # Do not silently save an incomplete dataset
    if maha.empty or yala.empty:
        raise ValueError(
            "One season has zero records. "
            "Check the Excel header rows and year formats."
        )

    sown_df = pd.concat(
        [maha, yala],
        ignore_index=True
    )

    # Standardize known district spelling differences
    district_mapping = {
        "KILLINOCHCHI": "KILINOCHCHI",
        "KILINOCHCHIYA": "KILINOCHCHI",
        "NUWARAELIYA": "NUWARA ELIYA",
        "MAHAWELI  'H'": "MAHAWELI 'H'",
        "MAHAWELI'H'": "MAHAWELI 'H'"
    }

    sown_df["district"] = sown_df["district"].replace(
        district_mapping
    )

    keys = [
        "district",
        "year",
        "season",
        "irrigation"
    ]

    duplicates = sown_df.duplicated(subset=keys).sum()

    print("\n===== VALIDATION =====")
    print("Total records:", len(sown_df))

    print("\nRecords by season:")
    print(sown_df["season"].value_counts())

    print("\nUnique districts:")
    print(sown_df["district"].nunique())

    print("\nDuplicate records:", duplicates)

    print("\nSuspicious year records:")
    print((sown_df["year"] == "2000/2021").sum())

    print("\nMissing values:")
    print(sown_df.isna().sum())

    if duplicates > 0:
        raise ValueError(
            f"Found {duplicates} duplicate records after cleaning."
        )

    sown_df = sown_df.sort_values(
        keys
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "paddy_sown_combined.csv"

    sown_df.to_csv(output_path, index=False)

    print("\n===== SAMPLE DATA =====")
    print(sown_df.head(10).to_string(index=False))

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, OSError, IndexError) as error:
        print(f"\nExtraction failed: {error}")
        raise SystemExit(1)
