import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_summary_report(df: pd.DataFrame):
    os.makedirs("outputs", exist_ok=True)

    crime_by_province = df.groupby("Province")["Cases"].sum().sort_values(ascending=False)
    crime_by_category = df.groupby("Category")["Cases"].sum().sort_values(ascending=False).head(10)
    crime_by_year = df.groupby("Year")["Cases"].sum().sort_index()

    print("\n=== Total Crime by Province ===")
    print(crime_by_province)

    print("\n=== Top 10 Crime Categories ===")
    print(crime_by_category)

    print("\n=== Crime Trend by Year ===")
    print(crime_by_year)

    crime_by_province.to_csv("outputs/crime_by_province.csv")
    crime_by_category.to_csv("outputs/top_crime_categories.csv")
    crime_by_year.to_csv("outputs/crime_by_year.csv")

    plt.figure(figsize=(10, 5))
    crime_by_year.plot(marker="o")
    plt.title("Crime Trend by Year")
    plt.ylabel("Cases")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig("outputs/crime_trend.png")
    plt.close()

    print("\nReport files saved in outputs/")


if __name__ == "__main__":
    df = pd.read_csv("clean_crime_data.csv")
    generate_summary_report(df)