
import pandas as pd

# 1. Load cleaned data
df = pd.read_csv("data/processed/paddy_yield_clean.csv")

print("Cleaned dataset records:", len(df))

# 2. Keep only administrative districts
df = df[df["area_type"] == "District"].copy()

print("After district selection:", len(df))

# 3. Keep overall district yield
df = df[df["irrigation"] == "Total"].copy()

print("After Total irrigation selection:", len(df))

# 4. Keep positive yields
df = df[df["yield_valid_for_model"]].copy()

print("After positive yield selection:", len(df))

# 5. Exclude unverified year labels
df = df[df["year"] != "2000/2021"].copy()

# 6. Validate unique observations
assert not df.duplicated(
    subset=["district", "year", "season"]
).any(), "Duplicate district-season-year records found!"

# 7. Save the modeling dataset
df.to_csv(
    "data/processed/paddy_yield_ml.csv",
    index=False
)

print("\nFinal ML records:", len(df))
print("Unique districts:", df["district"].nunique())
print("\nRecords by season:")
print(df["season"].value_counts())
