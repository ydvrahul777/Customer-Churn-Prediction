import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model, scaler, and encoders
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('robust_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Define features
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = ['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']

all_features = cat_features + num_features   # Order matters

def predict_churn(input_data):
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col in cat_features:
        if col in encoders:
            le = encoders[col]
            val = input_df[col].iloc[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = -1  # Handle unknown category
            input_df[col] = input_df[col].astype(object)

    # Scale numerical features
    input_df[num_features] = scaler.transform(input_df[num_features])

    # Ensure correct column order
    input_df = input_df[all_features]

    # Prediction
    prediction = model.predict(input_df)[0]
    return "Customer will Churn" if prediction == 1 else "Customer will NOT Churn"

def main():
    st.title("🔍 Customer Churn Prediction App")
    st.markdown("This app predicts whether a customer will churn based on their details.")

    # Numerical Inputs
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, step=1.0)
    total = st.number_input("Total Charges ($)", min_value=0.0, step=1.0)

    # Categorical Inputs
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    for col in cat_features:
        input_data[col] = st.selectbox(col, encoders[col].classes_)

    # Prediction button
    if st.button("Predict"):
        result = predict_churn(input_data)
        st.success(f"### Prediction: **{result}**")

if __name__ == "__main__":
    main()
