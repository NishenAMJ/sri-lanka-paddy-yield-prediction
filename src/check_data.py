import pandas as pd

df = pd.read_csv("data/processed/paddy_yield_combined.csv")

print(df.head())

print("\n=== HIGH YIELDS ===")

high_yields = df[df["average_yield_kg_ha"] > 10000]

print(high_yields.to_string(index=False))

print("\n=== ZERO OR NEGATIVE YIELDS ===")

zero_yields = df[df["average_yield_kg_ha"] <= 0]

print(zero_yields.to_string(index=False))

print("\n=== SUSPICIOUS YEARS ===")

suspicious_years = df[
    df["year"].astype(str).str.contains(
        "2000/2021",
        regex=False
    )
]

print(suspicious_years.head(10).to_string(index=False))
print("Total suspicious year records:", len(suspicious_years))

print("\n=== DISTRICT COUNTS ===")

district_counts = (
    df.groupby("district")
    .size()
    .sort_values()
)

print(district_counts.to_string())

print("\n=== IRRIGATION COUNTS ===")

print(df["irrigation"].value_counts())