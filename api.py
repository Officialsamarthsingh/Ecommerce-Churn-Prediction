from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import numpy as np
import joblib
import pickle

# Load model and scaler

scaler = joblib.load('model/scaler.pkl')

model = joblib.load('' \
'model/xgboost_churn_model.pkl')

app = FastAPI(title="Customer Churn Prediction API")

# Input schema — user sends readable values, not one-hot encoded columns
class CustomerInput(BaseModel):
    Tenure: float
    WarehouseToHome: float
    NumberOfDeviceRegistered: int
    SatisfactionScore: int
    NumberOfAddress: int
    Complain: Literal[0, 1]
    DaySinceLastOrder: float
    CashbackAmount: float
    PreferedOrderCat: Literal["Fashion", "Grocery", "Laptop & Accessory", "Mobile", "Others"]
    MaritalStatus: Literal["Divorced", "Married", "Single"]


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerInput):

    # One-hot encode PreferedOrderCat
    all_cats = ["Fashion", "Grocery", "Laptop & Accessory", "Mobile", "Others"]
    cat_encoded = [1 if data.PreferedOrderCat == cat else 0 for cat in all_cats]

    # One-hot encode MaritalStatus
    all_marital = ["Divorced", "Married", "Single"]
    marital_encoded = [1 if data.MaritalStatus == m else 0 for m in all_marital]

    # Log transform + build input array
    # Same preprocessing applied during training
    input_data = np.array([[
        np.log1p(data.Tenure),
        np.log1p(data.WarehouseToHome),
        data.NumberOfDeviceRegistered,
        data.SatisfactionScore,
        data.NumberOfAddress,
        data.Complain,
        np.log1p(data.DaySinceLastOrder),
        np.log1p(data.CashbackAmount),
        *cat_encoded,
        *marital_encoded
    ]])

    # Scale using the same fitted scaler from training
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if prediction == 1 else "Low"
    }