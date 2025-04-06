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

# Define feature names
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = ['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

all_features = cat_features + num_features

def predict_churn(input_dict):
    input_df = pd.DataFrame([input_dict])

    # Encode categorical features
    for col in cat_features:
        if col in encoders:
            le = encoders[col]
            if input_df[col][0] in le.classes_:
                input_df[col] = le.transform([input_df[col][0]])
            else:
                input_df[col] = -1  # unknown category

    # Convert numeric values to float and scale them
    input_df[num_features] = input_df[num_features].astype(float)
    input_df[num_features] = scaler.transform(input_df[num_features])

    # Ensure correct feature order
    input_df = input_df[all_features]

    # Make prediction
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        return "Yes — The customer is likely to churn."
    else:
        return "No — The customer is likely to stay."

def main():
    st.title("📉 Customer Churn Predictor")
    st.subheader("Fill in the customer details below:")

    # Numeric inputs
    tenure = st.number_input("Tenure (months)", min_value=0)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0)

    # Categorical inputs
    inputs = {}
    for col in cat_features:
        options = encoders[col].classes_.tolist()
        inputs[col] = st.selectbox(col.replace('_', ' '), options)

    # Add numerical inputs
    inputs['tenure'] = tenure
    inputs['MonthlyCharges'] = monthly_charges
    inputs['TotalCharges'] = total_charges

    # Predict button
    if st.button("Predict Churn"):
        result = predict_churn(inputs)
        st.success(f"🧾 Prediction Result: **{result}**")

if __name__ == '__main__':
    main()
