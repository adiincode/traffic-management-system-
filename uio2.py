import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Traffic Volume Predictor",
    page_icon="🚦",
    layout="wide"
)

st.title("Aditya's Traffic Volume Prediction System")
st.write("Predict Traffic Volume using Weather Conditions")

uploaded_file = st.file_uploader(
    "Upload Traffic Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Remove holiday column if present
    if "holiday" in df.columns:
        df.drop(columns=["holiday"], inplace=True)

    # Convert datetime
    df["date_time"] = pd.to_datetime(
        df["date_time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    df.dropna(inplace=True)

    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month
    df["day"] = df["date_time"].dt.day
    df["hour"] = df["date_time"].dt.hour
    df["dayofweek"] = df["date_time"].dt.dayofweek

    df.drop(columns=["date_time"], inplace=True)

    # One Hot Encoding
    df = pd.get_dummies(
        df,
        columns=["weather_main", "weather_description"],
        drop_first=True
    )

    X = df.drop("traffic_volume", axis=1)
    y = df["traffic_volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    st.success(" Model Trained Successfully")

    st.sidebar.header("Enter Input Values")

    temp = st.sidebar.number_input("Temperature", value=290.0)

    rain = st.sidebar.number_input("Rain (1h)", value=0.0)

    snow = st.sidebar.number_input("Snow (1h)", value=0.0)

    clouds = st.sidebar.slider(
        "Clouds (%)",
        0,
        100,
        50
    )

    year = st.sidebar.number_input(
        "Year",
        value=2016
    )

    month = st.sidebar.slider(
        "Month",
        1,
        12,
        6
    )

    day = st.sidebar.slider(
        "Day",
        1,
        31,
        15
    )

    hour = st.sidebar.slider(
        "Hour",
        0,
        23,
        12
    )

    dayofweek = st.sidebar.slider(
        "Day Of Week",
        0,
        6,
        2
    )

    input_data = pd.DataFrame(
        np.zeros((1, len(X.columns))),
        columns=X.columns
    )

    input_data["temp"] = temp
    input_data["rain_1h"] = rain
    input_data["snow_1h"] = snow
    input_data["clouds_all"] = clouds
    input_data["year"] = year
    input_data["month"] = month
    input_data["day"] = day
    input_data["hour"] = hour
    input_data["dayofweek"] = dayofweek

    if st.button("Predict Traffic Volume"):

        prediction = model.predict(input_data)

        st.metric(
            "Predicted Traffic Volume",
            f"{prediction[0]:.0f}"
        )

        st.balloons()

    st.subheader("Dataset Statistics")
    st.write(df.describe())

    st.subheader("Correlation Matrix")
    st.dataframe(df.corr(numeric_only=True))

else:
    st.info("Please upload the traffic dataset.")