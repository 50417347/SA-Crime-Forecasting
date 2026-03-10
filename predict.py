import joblib
import pandas as pd


def reverse_map(mapping: dict) -> dict:
    return {v: k for k, v in mapping.items()}


def predict_cases(province: str, station: str, category: str, year: int):
    model = joblib.load("crime_model.pkl")
    encoders = joblib.load("crime_encoders.pkl")
    df_model = pd.read_csv("crime_model_data.csv")

    province_to_code = reverse_map(encoders["province_map"])
    station_to_code = reverse_map(encoders["station_map"])
    category_to_code = reverse_map(encoders["category_map"])

    if province not in province_to_code:
        raise ValueError(f"Unknown province: {province}")
    if station not in station_to_code:
        raise ValueError(f"Unknown station: {station}")
    if category not in category_to_code:
        raise ValueError(f"Unknown category: {category}")

    history = df_model[
        (df_model["Province"] == province) &
        (df_model["Station"] == station) &
        (df_model["Category"] == category)
    ].sort_values("Year")

    if len(history) < 2:
        raise ValueError("Not enough history for prediction.")

    last_row = history.iloc[-1]
    prev_row = history.iloc[-2]

    input_df = pd.DataFrame([{
        "Province_code": province_to_code[province],
        "Station_code": station_to_code[station],
        "Category_code": category_to_code[category],
        "Year": year,
        "lag_1": last_row["Cases"],
        "lag_2": prev_row["Cases"],
        "rolling_mean_2": (last_row["Cases"] + prev_row["Cases"]) / 2
    }])

    prediction = model.predict(input_df)[0]
    return round(float(prediction), 2)


if __name__ == "__main__":
    pred = predict_cases(
        province="Western Cape",
        station="Cape Town Central",
        category="All theft not mentioned elsewhere",
        year=2016
    )
    print("Predicted cases:", round(pred))