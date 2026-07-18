import streamlit as st
import pandas as pd
import os
import joblib

#Load model
model=joblib.load("car_price.pkl")
columns=joblib.load("columns.pkl")
make_model=joblib.load("make_model.pkl")
scaler=None
if os.path.exists('scaler.pkl'):
    scaler=joblib.load("scaler.pkl")
    
#Streamlit page
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout='wide'
)
st.title("🚗Car Price Prediction")
st.write("Choose car Model")

#user input
col1, col2 = st.columns(2)
with col1:
    make = st.selectbox("Make", sorted(make_model.keys()))
with col2:
    select_model = st.selectbox('Model', sorted(make_model[make]))
st.write("Enter Car Details")
col1,col2=st.columns(2)
with col1:
    year=st.number_input(
        "Manufacturing Year",
        min_value=1988,
        max_value=2026,
        value=1988
    )
    kilometer = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=87150
    )
    fuel_type = st.selectbox(
        "Fuel Type",
        [
            "Petrol",
            "Diesel"
        ]
    )
    transmission = st.selectbox(
        "Transmission",
        [
            "Manual",
            "Automatic"
        ]
    )
    locations=[]
    for col in columns:
        if col.startswith('Location_'):
            locations.append(col.replace("Location_",""))
    location=st.selectbox(
        "Location",
        sorted(locations)
    )
with col2:
    color=st.text_input('Color','Grey')
    owner = st.selectbox(
        "Owner",
        [
            "First",
            "Second"
        ]
    )
    drivetrain = st.selectbox(
        "Drivetrain",
        [
            "FWD",
            "RWD",
            "AWD"
        ]
    )
    seating_capacity = st.number_input(
        "Seating Capacity",
        min_value=5,
        max_value=7,
        value=5
    )
    fuel_tank_capacity = st.number_input(
        "Fuel Tank Capacity (L)",
        value=35.0
    )
predict = st.button("Predict Price")

if predict:
    # Create a dataframe with numerical features
    input_df = pd.DataFrame([{
        "Year": year,
        "Kilometer": kilometer,
        "Seating_Capacity": seating_capacity,
        "Fuel_Tank_Capacity": fuel_tank_capacity
    }])
    # Create all training columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    # One-hot encode categorical values
    for value in [
        "Make_" + make,
        "Model_" + select_model,
        "Fuel_Type_" + fuel_type,
        "Transmission_" + transmission,
        "Location_" + location,
        "Color_" + color,
        "Owner_" + owner,
        "Drivetrain_" + drivetrain,
    ]:
        if value in input_df.columns:
            input_df[value] = 1
    # Arrange columns exactly like training
    input_df = input_df[columns]
    # Scale if scaler exists
    if scaler is not None:
        input_df = scaler.transform(input_df)
    # Predict
    prediction = model.predict(input_df)
    st.success(f"🚗 Predicted Car Price: ₹ {prediction[0]:,.2f}")