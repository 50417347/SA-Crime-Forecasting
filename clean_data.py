import pandas as pd
from load_data import load_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Province", "Station", "Category"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_long = df.melt(
        id_vars=["Province", "Station", "Category"],
        var_name="Year",
        value_name="Cases"
    )

    df_long["Province"] = df_long["Province"].astype(str).str.strip()
    df_long["Station"] = df_long["Station"].astype(str).str.strip()
    df_long["Category"] = df_long["Category"].astype(str).str.strip()

    df_long["Year"] = df_long["Year"].astype(str).str[:4]
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")

    df_long["Cases"] = pd.to_numeric(df_long["Cases"], errors="coerce")

    df_long = df_long.dropna(subset=["Year", "Cases"])
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["Cases"] = df_long["Cases"].astype(int)

    df_long = df_long.sort_values(["Province", "Station", "Category", "Year"]).reset_index(drop=True)
    return df_long


if __name__ == "__main__":
    df = load_data("crime_stats.xlsx")
    clean_df = clean_data(df)
    print(clean_df.head())
    clean_df.to_csv("clean_crime_data.csv", index=False)
    print("Saved clean_crime_data.csv")