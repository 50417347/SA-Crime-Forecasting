import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="South African Crime Dashboard", layout="wide")


@st.cache_data
def load_clean_data():
    return pd.read_csv("clean_crime_data.csv")


@st.cache_resource
def load_model():
    model = joblib.load("crime_model.pkl")
    encoders = joblib.load("crime_encoders.pkl")
    model_data = pd.read_csv("crime_model_data.csv")
    return model, encoders, model_data


def reverse_map(mapping: dict) -> dict:
    return {v: k for k, v in mapping.items()}


def predict_cases(model, encoders, df_model, province, station, category, year):
    province_to_code = reverse_map(encoders["province_map"])
    station_to_code = reverse_map(encoders["station_map"])
    category_to_code = reverse_map(encoders["category_map"])

    if province not in province_to_code:
        return None, f"Unknown province: {province}"
    if station not in station_to_code:
        return None, f"Unknown station: {station}"
    if category not in category_to_code:
        return None, f"Unknown category: {category}"

    history = df_model[
        (df_model["Province"] == province)
        & (df_model["Station"] == station)
        & (df_model["Category"] == category)
    ].sort_values("Year")

    if len(history) < 2:
        return None, "Not enough history for prediction."

    last_row = history.iloc[-1]
    prev_row = history.iloc[-2]

    input_df = pd.DataFrame(
        [
            {
                "Province_code": province_to_code[province],
                "Station_code": station_to_code[station],
                "Category_code": category_to_code[category],
                "Year": year,
                "lag_1": last_row["Cases"],
                "lag_2": prev_row["Cases"],
                "rolling_mean_2": (last_row["Cases"] + prev_row["Cases"]) / 2,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    return int(round(prediction)), None


def get_risk_level(reference_df, category, predicted_cases):
    category_data = reference_df[reference_df["Category"] == category]["Cases"]

    if category_data.empty:
        return "Unknown", "No reference data available."

    q1 = category_data.quantile(0.25)
    q2 = category_data.quantile(0.50)
    q3 = category_data.quantile(0.75)

    if predicted_cases <= q1:
        return "Low", f"Prediction is in the lower range for {category}."
    elif predicted_cases <= q2:
        return "Medium", f"Prediction is around the typical range for {category}."
    elif predicted_cases <= q3:
        return "High", f"Prediction is above average for {category}."
    else:
        return "Very High", f"Prediction is in the upper range for {category}."


def build_hotspot_map(map_df: pd.DataFrame):
    province_cases = (
        map_df.groupby("Province", as_index=False)["Cases"]
        .sum()
        .sort_values("Cases", ascending=False)
    )

    province_coords = {
        "Western Cape": {"lat": -33.93, "lon": 18.42},
        "Eastern Cape": {"lat": -32.30, "lon": 26.42},
        "Northern Cape": {"lat": -28.74, "lon": 24.76},
        "Free State": {"lat": -29.12, "lon": 26.21},
        "KwaZulu-Natal": {"lat": -29.85, "lon": 31.02},
        "Kwazulu/Natal": {"lat": -29.85, "lon": 31.02},
        "North West": {"lat": -25.67, "lon": 27.24},
        "Gauteng": {"lat": -26.20, "lon": 28.04},
        "Mpumalanga": {"lat": -25.47, "lon": 30.98},
        "Limpopo": {"lat": -23.90, "lon": 29.45},
    }

    province_cases["lat"] = province_cases["Province"].map(
        lambda x: province_coords.get(x, {}).get("lat")
    )
    province_cases["lon"] = province_cases["Province"].map(
        lambda x: province_coords.get(x, {}).get("lon")
    )

    province_cases = province_cases.dropna(subset=["lat", "lon"])

    fig = px.scatter_geo(
        province_cases,
        lat="lat",
        lon="lon",
        size="Cases",
        color="Cases",
        hover_name="Province",
        hover_data={"Cases": True, "lat": False, "lon": False},
        title="Crime Hotspot Map by Province",
        projection="natural earth",
        size_max=45,
    )

    fig.update_geos(
        scope="africa",
        center={"lat": -29.0, "lon": 24.0},
        projection_scale=6.5,
        showcountries=True,
        countrycolor="black",
        showsubunits=True,
        subunitcolor="gray",
        showland=True,
        landcolor="rgb(243,243,243)",
    )

    fig.update_layout(margin={"r": 20, "t": 50, "l": 20, "b": 20}, height=500)

    return fig


# Load data
df = load_clean_data()
model, encoders, model_data = load_model()

# Sidebar
st.sidebar.title("Filters")

province = st.sidebar.selectbox(
    "Province",
    ["All"] + sorted(df["Province"].dropna().unique().tolist()),
)

if province == "All":
    category_options = ["All"] + sorted(df["Category"].dropna().unique().tolist())
else:
    category_options = ["All"] + sorted(
        df[df["Province"] == province]["Category"].dropna().unique().tolist()
    )

category = st.sidebar.selectbox("Category", category_options)

year_min = int(df["Year"].min())
year_max = int(df["Year"].max())

selected_year_range = st.sidebar.slider(
    "Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This dashboard analyzes South African crime statistics from 2005 to 2015, "
    "shows trends and hotspots, and uses a machine learning model to forecast "
    "future crime cases from historical patterns."
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Absolom Muzambi, PhD student at UNISA")

# Filter data
filtered_df = df.copy()

if province != "All":
    filtered_df = filtered_df[filtered_df["Province"] == province]

if category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == category]

filtered_df = filtered_df[
    (filtered_df["Year"] >= selected_year_range[0])
    & (filtered_df["Year"] <= selected_year_range[1])
]

# Title
st.title("South African Crime Analysis Dashboard")
st.caption("Interactive analysis, hotspot detection, prediction, and risk scoring")
st.markdown("**Developed by Absolom Muzambi, PhD student at UNISA**")

# KPI section
total_cases = int(filtered_df["Cases"].sum()) if not filtered_df.empty else 0

if not filtered_df.empty:
    top_station = (
        filtered_df.groupby("Station")["Cases"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )
    top_category = (
        filtered_df.groupby("Category")["Cases"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )
    avg_cases_per_year = int(filtered_df.groupby("Year")["Cases"].sum().mean())
else:
    top_station = "N/A"
    top_category = "N/A"
    avg_cases_per_year = 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cases", f"{total_cases:,}")
col2.metric("Top Station", top_station)
col3.metric("Top Category", top_category)
col4.metric("Average Cases per Year", f"{avg_cases_per_year:,}")

# Main charts
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Crime Trend by Year")
    if not filtered_df.empty:
        year_counts = filtered_df.groupby("Year")["Cases"].sum().sort_index()
        st.line_chart(year_counts)
    else:
        st.info("No data available for the selected filters.")

with right_col:
    if province == "All":
        st.subheader("Crime by Province")
        if not filtered_df.empty:
            province_counts = (
                filtered_df.groupby("Province")["Cases"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(province_counts)
        else:
            st.info("No data available for the selected filters.")
    else:
        st.subheader("Top Stations in Selected Province")
        if not filtered_df.empty:
            station_chart = (
                filtered_df.groupby("Station")["Cases"]
                .sum()
                .sort_values(ascending=False)
                .head(15)
            )
            st.bar_chart(station_chart)
        else:
            st.info("No data available for the selected filters.")

# Embedded hotspot map
st.subheader("Crime Hotspot Map")
if not filtered_df.empty:
    hotspot_fig = build_hotspot_map(filtered_df)
    st.plotly_chart(hotspot_fig, width="stretch")
else:
    st.info("No data available for the selected filters.")

# Secondary charts
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Top 10 Crime Categories")
    if not filtered_df.empty:
        top_categories = (
            filtered_df.groupby("Category")["Cases"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(top_categories)
    else:
        st.info("No category data available.")

with col_b:
    st.subheader("Top 10 Stations by Cases")
    if not filtered_df.empty:
        top_stations = (
            filtered_df.groupby("Station")["Cases"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={"Cases": "Total Cases"})
        )
        st.dataframe(top_stations, width="stretch")
    else:
        st.info("No station data available.")

# Prediction section
st.subheader("Predict Future Crime Cases")

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    pred_province = st.selectbox(
        "Select Province for Prediction",
        sorted(df["Province"].dropna().unique().tolist()),
        key="pred_province",
    )

    available_stations = sorted(
        df[df["Province"] == pred_province]["Station"].dropna().unique().tolist()
    )
    pred_station = st.selectbox(
        "Select Station",
        available_stations,
        key="pred_station",
    )

with pred_col2:
    available_categories = sorted(
        df[
            (df["Province"] == pred_province)
            & (df["Station"] == pred_station)
        ]["Category"].dropna().unique().tolist()
    )
    pred_category = st.selectbox(
        "Select Crime Category",
        available_categories,
        key="pred_category",
    )

    pred_year = st.number_input(
        "Prediction Year",
        min_value=int(df["Year"].max()) + 1,
        max_value=2035,
        value=int(df["Year"].max()) + 1,
        step=1,
    )

if st.button("Predict Cases"):
    prediction, error = predict_cases(
        model,
        encoders,
        model_data,
        pred_province,
        pred_station,
        pred_category,
        pred_year,
    )

    if error:
        st.error(error)
    else:
        risk_level, risk_message = get_risk_level(df, pred_category, prediction)

        st.success(
            f"Predicted cases for {pred_year} in {pred_station} "
            f"({pred_category}): {prediction:,}"
        )

        if risk_level == "Low":
            st.info(f"Risk Level: {risk_level} — {risk_message}")
        elif risk_level in ["Medium", "High"]:
            st.warning(f"Risk Level: {risk_level} — {risk_message}")
        else:
            st.error(f"Risk Level: {risk_level} — {risk_message}")

# Download section
st.subheader("Download Filtered Data")
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered data as CSV",
    data=csv_data,
    file_name="filtered_crime_data.csv",
    mime="text/csv",
)

# Raw data preview
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df, width="stretch")
