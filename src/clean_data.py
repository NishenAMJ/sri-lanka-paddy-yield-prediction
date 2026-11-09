
import pandas as pd
from pathlib import Path

# 1. Define input and output paths
INPUT_FILE = Path("data/processed/paddy_yield_combined.csv")
OUTPUT_FILE = Path("data/processed/paddy_yield_clean.csv")

# 2. Load the dataset
df = pd.read_csv(INPUT_FILE)

print("Original records:", len(df))

# 3. Standardize district names
district_mapping = {
    "KILLINOCHCHI": "KILINOCHCHI",
    "MAHAWELI  'H'": "MAHAWELI 'H'"
}

df["district"] = df["district"].replace(district_mapping)

# 4. Flag suspicious year labels
suspicious_years = df["year"] == "2000/2021"

print("Suspicious year records:", suspicious_years.sum())

# Correct the label after confirming it in the source workbook
# df["year"] = df["year"].replace({
#     "2000/2021": "2020/2021"
# })

# 5. Classify geographic records
special_areas = ["MAHAWELI 'H'", "UDAWALAWE"]

df["area_type"] = "District"

df.loc[df["district"] == "SRI LANKA", "area_type"] = "National"

df.loc[
    df["district"].isin(special_areas),
    "area_type"
] = "Special Area"

# 6. Flag nonpositive yield values
df["yield_valid_for_model"] = df["average_yield_kg_ha"] > 0

# 7. Check duplicate records
duplicates = df.duplicated(
    subset=["district", "year", "season", "irrigation"]
).sum()

print("Duplicate records:", duplicates)

# 8. Save the cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned records:", len(df))
print("\nArea types:")
print(df["area_type"].value_counts())

print("\nNonpositive yield records:")
print((~df["yield_valid_for_model"]).sum())

print("\nCleaning completed!")
