
import pandas as pd
from pathlib import Path

file_path = Path(
    "data/raw/Production-Maha_Season-1979-2025.xlsx"
)

df = pd.read_excel(file_path, header=None)

print("===== WORKBOOK SHAPE =====")
print(df.shape)

print("\n===== FIRST 8 ROWS, FIRST 12 COLUMNS =====")
print(df.iloc[:8, :12].to_string(index=True, header=True))

print("\n===== HEADER ROWS, LAST 15 COLUMNS =====")
print(df.iloc[:4, -15:].to_string(index=True, header=True))

print("\n===== AMPARA ROW =====")

matches = df.apply(
    lambda col: col.astype(str)
    .str.strip()
    .str.upper()
    .eq("AMPARA")
).any(axis=1)

for row_index in df.index[matches]:
    print("Row index:", row_index)
    print("First 12 values:")
    print(df.iloc[row_index, :12].tolist())

    print("Last 15 values:")
    print(df.iloc[row_index, -15:].tolist())
