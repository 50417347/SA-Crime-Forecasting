import pandas as pd
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        engine = "openpyxl" if path.suffix.lower() == ".xlsx" else "xlrd"
        df = pd.read_excel(path, engine=engine)
    else:
        raise ValueError("Unsupported file type. Use .csv, .xlsx, or .xls")

    df.columns = df.columns.astype(str).str.strip()
    return df


if __name__ == "__main__":
    df = load_data("crime_stats.xlsx")
    print(df.head())
    print(df.columns.tolist())