import streamlit as st
import numpy as np 
import joblib 

model = joblib.load("E:/Sam_Analysis/09_Project_Customer_Churn_Prediction_Model/Churn_Prediction_Model/model/xgboost_churn_model.pkl")
scaler = joblib.load("E:/Sam_Analysis/09_Project_Customer_Churn_Prediction_Model/Churn_Prediction_Model/model/scaler.pkl")

st.title("CUSTOMER CHURN PREDICTION")
st.write("Fill the the customer details below.")

#Numeric Input
tenure = st.slider("Tenure (months)",0,60,12)
warehouse = st.slider("Warehouse to Home Distance(Km)",1,130,15)
num_devices = st.slider("Number of Devices Registered",1,6,3)
satisfaction = st.slider("Satisfaction Score",1,5,3)
num_address = st.slider("Number of Address",0,60,12)
complain = st.radio("Has Complained?",[0,1],
                    format_func=lambda x: "Yes" if x==1 else "No")
days_orders = st.slider("Days Since Last Orders",0,30,5)
cashback = st.number_input("Cashback Amount",
                    min_value=100.0, max_value=350.0,value=150.0)

#Single-select inputs
order_cat = st.selectbox("Preferred Order Category",
                    ["Fashion","Grocery","Laptop & Accessory","Mobile","Others"])
marital = st.selectbox("Marital Status",
                    ["Divorced","Married","Single"])

if st.button("Predict"):
    #One-hot encode
    all_cats = ["Fashion","Grocery","Laptop & Accessory","Mobile","Others"]
    all_marital = ["Divorced","Married","Single"]
    cat_encoded = [1 if order_cat == c else 0 for c in all_cats]
    marital_encoded = [1 if marital == m else 0 for m in all_marital]

    # Log transform + bulid input array
    input_data = np.array([[
        np.log1p(tenure),
        np.log1p(warehouse),
        num_devices,
        satisfaction,
        num_address,
        complain,
        np.log1p(days_orders),
        np.log1p(cashback),
        *cat_encoded,
        *marital_encoded
    ]])    

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.markdown("---")
    if prediction==1:
        st.error(f"HIGH CHURN RISK : {probability*100:.1F}% PROBILITY")
    else:
        st.success(f"LOW CHURN RISK : {probability*100:.1F}% PROBILITY")
