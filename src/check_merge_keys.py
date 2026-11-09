
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

files = {
    "Yield": "paddy_yield_clean.csv",
    "Production": "paddy_production_combined.csv",
    "Harvested": "paddy_harvested_combined.csv",
    "Sown": "paddy_sown_combined.csv"
}

for name, filename in files.items():
    df = pd.read_csv(DATA_DIR / filename, dtype={"year": str})

    print(f"\n===== {name} =====")
    print("Records:", len(df))
    print("Columns:", df.columns.tolist())
    print("Districts:", sorted(df["district"].dropna().unique()))
    print("Seasons:", df["season"].value_counts().to_dict())
    print("Year examples:", df["year"].dropna().unique()[:5])

    keys = ["district", "year", "season"]

    if "irrigation" in df.columns:
        keys.append("irrigation")

    print("Duplicate keys:", df.duplicated(subset=keys).sum())
