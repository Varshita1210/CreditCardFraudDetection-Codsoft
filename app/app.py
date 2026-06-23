import streamlit as st
import joblib
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "fraud_model.pkl")

model = joblib.load(model_path)



st.title("💳 Credit Card Fraud Detection")

st.write("Enter Transaction Details")

trans_time = st.number_input("Transaction Time", value=514)
merchant = st.number_input("Merchant", value=8)
category = st.number_input("Category", value=4)
amt = st.number_input("Amount", value=4.97)

gender = st.number_input("Gender", value=1)
city = st.number_input("City", value=241)
state = st.number_input("State", value=39)

zip_code = st.number_input("Zip", value=29209)

lat = st.number_input("Latitude", value=36.0788)
longi = st.number_input("Longitude", value=-81.1781)

city_pop = st.number_input("City Population", value=3495)

job = st.number_input("Job", value=275)

dob = st.number_input("DOB Encoded", value=514)

unix_time = st.number_input("Unix Time", value=1325376018)

merch_lat = st.number_input("Merchant Latitude", value=36.0112)

merch_long = st.number_input("Merchant Longitude", value=-82.0483)
if st.button("Load Fraud Example"):
    st.error("🚨 Known Fraud Transaction Loaded")
    

if st.button("Predict Fraud"):

    data = np.array([[

        trans_time,
        merchant,
        category,
        amt,
        gender,
        city,
        state,
        zip_code,
        lat,
        longi,
        city_pop,
        job,
        dob,
        unix_time,
        merch_lat,
        merch_long

    ]])

    prediction = model.predict(data)

    if prediction[0] == 0:
        st.success("✅ Legitimate Transaction")
    else:
        st.error("🚨 Fraudulent Transaction")