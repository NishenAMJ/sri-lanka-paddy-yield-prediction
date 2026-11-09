
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

df = pd.read_csv(
    DATA_DIR / "paddy_ml_features_v2.csv",
    dtype={"year": str}
)

train_df = df[df["season_year"] < 2015].copy()

val_df = df[
    (df["season_year"] >= 2015) &
    (df["season_year"] < 2020)
].copy()

test_df = df[df["season_year"] >= 2020].copy()

print("===== V2 DATASET SPLIT =====")

for name, dataset in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    print(f"\n{name}:")
    print("Records:", len(dataset))
    print(
        "Years:",
        dataset["season_year"].min(),
        "-",
        dataset["season_year"].max()
    )
    print("Missing previous yield:",
          dataset["previous_yield"].isna().sum())
    print("Missing 3-season average:",
          dataset["yield_3season_avg"].isna().sum())

train_df.to_csv(
    DATA_DIR / "paddy_train_v2.csv",
    index=False
)

val_df.to_csv(
    DATA_DIR / "paddy_validation_v2.csv",
    index=False
)

test_df.to_csv(
    DATA_DIR / "paddy_test_v2.csv",
    index=False
)

print("\nV2 datasets saved successfully.")
