# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and preprocessing tools
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Define features
cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod']

num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

all_features = cat_features + num_features

# Prediction function
def predict_churn(input_dict):
    df = pd.DataFrame([input_dict])

    # Encode categorical columns
    for col in cat_features:
        if col in encoders:
            if df[col][0] in encoders[col].classes_:
                df[col] = encoders[col].transform([df[col][0]])
            else:
                df[col] = -1  # unknown category
        df[col] = df[col].astype('object')  # Convert back to object after encoding

    # Scale numerical features
    df[num_features] = scaler.transform(df[num_features])

    # Reorder columns
    df = df[all_features]

    # Make prediction
    prediction = model.predict(df)[0]
    return "Yes, the customer will churn." if prediction == 1 else "No, the customer will not churn."


# Streamlit UI
def main():
    st.title("Customer Churn Predictor")
    st.subheader("Provide customer details to check churn probability.")

    input_data = {}

    # Categorical input fields
    for feature in cat_features:
        input_data[feature] = st.selectbox(f"{feature}", encoders[feature].classes_)

    # Numerical input fields
    input_data['tenure'] = st.number_input("Tenure (months)", min_value=0, step=1)
    input_data['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0, format="%.2f")
    input_data['TotalCharges'] = st.number_input("Total Charges", min_value=0.0, format="%.2f")

    if st.button("Predict Churn"):
        result = predict_churn(input_data)
        st.success(result)


if __name__ == "__main__":
    main()
