import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Province", "Station", "Category", "Year"]).reset_index(drop=True)

    group_cols = ["Province", "Station", "Category"]

    df["lag_1"] = df.groupby(group_cols)["Cases"].shift(1)
    df["lag_2"] = df.groupby(group_cols)["Cases"].shift(2)

    df["rolling_mean_2"] = df.groupby(group_cols)["Cases"].transform(
        lambda x: x.shift(1).rolling(2).mean()
    )

    df = df.dropna().reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame):
    df_model = add_time_features(df)

    province_cat = df_model["Province"].astype("category")
    station_cat = df_model["Station"].astype("category")
    category_cat = df_model["Category"].astype("category")

    df_model["Province_code"] = province_cat.cat.codes
    df_model["Station_code"] = station_cat.cat.codes
    df_model["Category_code"] = category_cat.cat.codes

    encoders = {
        "province_map": dict(enumerate(province_cat.cat.categories)),
        "station_map": dict(enumerate(station_cat.cat.categories)),
        "category_map": dict(enumerate(category_cat.cat.categories)),
    }

    X = df_model[
        [
            "Province_code",
            "Station_code",
            "Category_code",
            "Year",
            "lag_1",
            "lag_2",
            "rolling_mean_2",
        ]
    ]
    y = df_model["Cases"]

    return X, y, encoders, df_model


def train_model(df: pd.DataFrame):
    X, y, encoders, df_model = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, encoders, mae, r2, df_model


if __name__ == "__main__":
    df = pd.read_csv("clean_crime_data.csv")

    model, encoders, mae, r2, df_model = train_model(df)

    joblib.dump(model, "crime_model.pkl", compress=3)
    joblib.dump(encoders, "crime_encoders.pkl", compress=3)
    df_model.to_csv("crime_model_data.csv", index=False)

    print("Model saved to crime_model.pkl")
    print("Encoders saved to crime_encoders.pkl")
    print("Model data saved to crime_model_data.csv")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")