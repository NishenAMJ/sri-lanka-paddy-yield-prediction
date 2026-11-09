
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

df = pd.read_csv(DATA_DIR / "paddy_train.csv")

train_df = df[df["season_year"] < 2015].copy()
val_df = df[
    (df["season_year"] >= 2015)
    & (df["season_year"] < 2020)
].copy()

print("===== TRAIN / VALIDATION SPLIT =====")
print("Training records:", len(train_df))
print("Validation records:", len(val_df))

print("\nTraining years:")
print(train_df["season_year"].min(), "-", train_df["season_year"].max())

print("\nValidation years:")
print(val_df["season_year"].min(), "-", val_df["season_year"].max())

train_df.to_csv(DATA_DIR / "paddy_train_inner.csv", index=False)
val_df.to_csv(DATA_DIR / "paddy_validation.csv", index=False)

print("\nValidation datasets saved.")
