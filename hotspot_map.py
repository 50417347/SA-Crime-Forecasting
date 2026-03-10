import os
import pandas as pd
import plotly.express as px


def generate_hotspot_map():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("clean_crime_data.csv")

    # Group total crime by province
    province_cases = (
        df.groupby("Province", as_index=False)["Cases"]
        .sum()
        .sort_values("Cases", ascending=False)
    )

    # Approximate centroid coordinates for SA provinces
    province_coords = {
        "Western Cape": {"lat": -33.93, "lon": 18.42},
        "Eastern Cape": {"lat": -32.30, "lon": 26.42},
        "Northern Cape": {"lat": -28.74, "lon": 24.76},
        "Free State": {"lat": -29.12, "lon": 26.21},
        "KwaZulu-Natal": {"lat": -29.85, "lon": 31.02},
        "Kwazulu/Natal": {"lat": -29.85, "lon": 31.02},   # handle alternate spelling
        "North West": {"lat": -25.67, "lon": 27.24},
        "Gauteng": {"lat": -26.20, "lon": 28.04},
        "Mpumalanga": {"lat": -25.47, "lon": 30.98},
        "Limpopo": {"lat": -23.90, "lon": 29.45},
    }

    province_cases["lat"] = province_cases["Province"].map(lambda x: province_coords.get(x, {}).get("lat"))
    province_cases["lon"] = province_cases["Province"].map(lambda x: province_coords.get(x, {}).get("lon"))

    province_cases = province_cases.dropna(subset=["lat", "lon"])

    fig = px.scatter_geo(
        province_cases,
        lat="lat",
        lon="lon",
        size="Cases",
        color="Cases",
        hover_name="Province",
        hover_data={"Cases": True, "lat": False, "lon": False},
        title="South African Crime Hotspot Map by Province",
        projection="natural earth",
        size_max=45,
    )

    fig.update_geos(
        scope="africa",
        center={"lat": -29.0, "lon": 24.0},
        projection_scale=6.5,
        showcountries=True,
        countrycolor="Black",
        showsubunits=True,
        subunitcolor="Gray",
        showland=True,
        landcolor="rgb(243, 243, 243)",
    )

    fig.update_layout(
        margin={"r": 20, "t": 50, "l": 20, "b": 20}
    )

    output_file = "outputs/crime_hotspot_map.html"
    fig.write_html(output_file)

    print(f"Hotspot map saved to: {output_file}")


if __name__ == "__main__":
    generate_hotspot_map()