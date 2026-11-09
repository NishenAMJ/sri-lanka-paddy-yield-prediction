
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

df = pd.read_csv(
    DATA_DIR / "paddy_ml_features.csv",
    dtype={"year": str}
)

features = [
    "district",
    "season",
    "season_year",
    "sown_extent_ha",
    "previous_yield",
    "yield_3season_avg"
]

target = "average_yield_kg_ha"

# Drop first observations without historical yield
df = df.dropna(
    subset=["previous_yield", "yield_3season_avg"]
).copy()

# Chronological split
train_df = df[df["season_year"] < 2020].copy()
test_df = df[df["season_year"] >= 2020].copy()

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("===== TRAIN / TEST SPLIT =====")
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nTraining years:")
print(
    train_df["season_year"].min(),
    "-",
    train_df["season_year"].max()
)

print("\nTesting years:")
print(
    test_df["season_year"].min(),
    "-",
    test_df["season_year"].max()
)

print("\nTraining features:")
print(X_train.head())

print("\nMissing training values:")
print(X_train.isna().sum())

print("\nMissing testing values:")
print(X_test.isna().sum())

train_df.to_csv(
    DATA_DIR / "paddy_train.csv",
    index=False
)

test_df.to_csv(
    DATA_DIR / "paddy_test.csv",
    index=False
)

print("\nTrain and test datasets saved.")
