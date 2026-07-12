import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model
# -------------------------------

model = joblib.load("traffic_prediction_model.pkl")

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Traffic Volume Predictor",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Aditya's Traffic Volume Prediction System")
st.write("Predict Traffic Volume using Weather Conditions")

st.markdown("---")

# -------------------------------
# Sidebar Inputs
# -------------------------------

st.sidebar.header("Enter Traffic Details")

holiday = st.sidebar.selectbox(
    "Holiday",
    ["None", "Christmas", "Columbus Day", "Veterans Day", "Washington Birthday"]
)

temp = st.sidebar.number_input(
    "Temperature",
    value=290.0
)

rain = st.sidebar.number_input(
    "Rain in last 1 hour",
    value=0.0
)

snow = st.sidebar.number_input(
    "Snow in last 1 hour",
    value=0.0
)

clouds = st.sidebar.slider(
    "Cloud Coverage (%)",
    0,
    100,
    40
)

weather_main = st.sidebar.selectbox(
    "Weather",
    [
        "Clear",
        "Clouds",
        "Rain",
        "Snow",
        "Mist",
        "Fog",
        "Drizzle",
        "Thunderstorm",
        "Haze"
    ]
)

weather_description = st.sidebar.text_input(
    "Weather Description",
    "sky is clear"
)

hour = st.sidebar.slider(
    "Hour",
    0,
    23,
    12
)

day = st.sidebar.slider(
    "Day",
    1,
    31,
    15
)

month = st.sidebar.slider(
    "Month",
    1,
    12,
    6
)

day_of_week = st.sidebar.slider(
    "Day of Week",
    0,
    6,
    2
)

is_weekend = 1 if day_of_week in [5, 6] else 0

# --------------------------------
# TEMPORARY Encoding
# --------------------------------

holiday_map = {
    "None":0,
    "Christmas":1,
    "Columbus Day":2,
    "Veterans Day":3,
    "Washington Birthday":4
}

weather_map = {
    "Clear":0,
    "Clouds":1,
    "Rain":2,
    "Snow":3,
    "Mist":4,
    "Fog":5,
    "Drizzle":6,
    "Thunderstorm":7,
    "Haze":8
}

holiday = holiday_map.get(holiday,0)
weather_main = weather_map.get(weather_main,0)

weather_description = 0

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Traffic Volume"):

    input_df = pd.DataFrame({

        "holiday":[holiday],
        "temp":[temp],
        "rain_1h":[rain],
        "snow_1h":[snow],
        "clouds_all":[clouds],
        "weather_main":[weather_main],
        "weather_description":[weather_description],
        "hour":[hour],
        "day":[day],
        "month":[month],
        "day_of_week":[day_of_week],
        "is_weekend":[is_weekend]

    })

    prediction = model.predict(input_df)

    st.success(f"🚗 Predicted Traffic Volume : {prediction[0]:.0f} Vehicles")

    st.balloons()

st.markdown("---")

st.info("Developed by Aditya Maurya ❤️")