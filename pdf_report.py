import os
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def generate_pdf_report():
    os.makedirs("outputs", exist_ok=True)

    province_file = "outputs/crime_by_province.csv"
    category_file = "outputs/top_crime_categories.csv"
    year_file = "outputs/crime_by_year.csv"
    chart_file = "outputs/crime_trend.png"
    pdf_file = "outputs/crime_summary_report.pdf"

    if not os.path.exists(province_file):
        raise FileNotFoundError(f"Missing file: {province_file}")
    if not os.path.exists(category_file):
        raise FileNotFoundError(f"Missing file: {category_file}")
    if not os.path.exists(year_file):
        raise FileNotFoundError(f"Missing file: {year_file}")

    crime_by_province = pd.read_csv(province_file)
    top_categories = pd.read_csv(category_file)
    crime_by_year = pd.read_csv(year_file)

    c = canvas.Canvas(pdf_file, pagesize=A4)
    width, height = A4

    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "South African Crime Analysis Report")
    y -= 25

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30

    # Summary section
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Summary")
    y -= 20

    total_cases = int(crime_by_year.iloc[:, 1].sum())
    top_province = crime_by_province.iloc[0, 0] if not crime_by_province.empty else "N/A"
    top_category = top_categories.iloc[0, 0] if not top_categories.empty else "N/A"

    c.setFont("Helvetica", 11)
    c.drawString(60, y, f"Total Cases: {total_cases:,}")
    y -= 18
    c.drawString(60, y, f"Top Province: {top_province}")
    y -= 18
    c.drawString(60, y, f"Top Crime Category: {top_category}")
    y -= 30

    # Top provinces
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Top 5 Provinces by Cases")
    y -= 20

    c.setFont("Helvetica", 10)
    for i, row in crime_by_province.head(5).iterrows():
        province = str(row.iloc[0])
        cases = int(row.iloc[1])
        c.drawString(60, y, f"{i+1}. {province}: {cases:,}")
        y -= 16

    y -= 15

    # Top categories
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Top 5 Crime Categories")
    y -= 20

    c.setFont("Helvetica", 10)
    for i, row in top_categories.head(5).iterrows():
        category = str(row.iloc[0])
        cases = int(row.iloc[1])
        c.drawString(60, y, f"{i+1}. {category}: {cases:,}")
        y -= 16

    y -= 20

    # Add chart if it exists
    if os.path.exists(chart_file):
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, "Crime Trend Chart")
        y -= 15

        img = ImageReader(chart_file)
        c.drawImage(img, 50, y - 220, width=500, height=220, preserveAspectRatio=True, mask='auto')
        y -= 240

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 30, "Generated automatically by the South African Crime Analysis Pipeline")

    c.save()
    print(f"PDF report saved to: {pdf_file}")


if __name__ == "__main__":
    generate_pdf_report()