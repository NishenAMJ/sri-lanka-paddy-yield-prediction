
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

for file in RAW_DIR.iterdir():
    if not file.is_file():
        continue

    print(f"\nFile: {file.name}")
    print(f"Size: {file.stat().st_size} bytes")

    try:
        if file.suffix == ".xlsx":
            workbook = pd.ExcelFile(file)

            print("Sheets:", workbook.sheet_names)

            for sheet in workbook.sheet_names:
                df = pd.read_excel(
                    file, sheet_name=sheet, header=None
                )
                print(f"\nSheet: {sheet}")
                print(df.head(8).to_string(index=False))

        elif file.suffix == ".xls":
            print("Legacy Excel file: inspect separately")

        elif file.suffix == ".html":
            tables = pd.read_html(str(file))
            print("Tables found:", len(tables))

            for i, table in enumerate(tables):
                print(f"\nTable {i}")
                print(table.head(8).to_string(index=False))

        elif file.suffix == ".pdf":
            print("PDF detected: table extraction needed")

        else:
            print("Unknown file format: manual inspection needed")

    except Exception as error:
        print("Could not inspect file:", error)
