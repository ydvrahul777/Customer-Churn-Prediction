import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and encoders
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('robust_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Define features
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = ['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

all_features = num_features + cat_features

# Prediction function
def predict_churn(input_data):
    input_df = pd.DataFrame([input_data])

    # Label encode categorical features
    for col in cat_features:
        if col in encoders:
            le = encoders[col]
            val = input_df[col].iloc[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = -1  # Handle unseen values
            input_df[col] = input_df[col].astype(object)

    num_df = input_df[num_features].astype(float)
    scaled_nums = pd.DataFrame(scaler.transform(num_df), columns=num_features)
    input_df[num_features] = scaled_nums[num_features]

    # Ensure column order
    input_df = input_df[all_features]

    # Make prediction
    prediction = model.predict(input_df)[0]
    return "Customer will Churn" if prediction == 1 else "Customer will NOT Churn"

# Streamlit App
def main():
    st.title("📊 Customer Churn Prediction App")
    st.markdown("Fill in customer details to predict if they will churn:")

    # Numerical Inputs
    tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100)
    monthly = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    total = st.number_input("Total Charges", min_value=0.0, step=1.0)

    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    # Dynamically collect categorical inputs based on encoders
    for col in cat_features:
        input_data[col] = st.selectbox(f"{col}", options=encoders[col].classes_)
    if st.button("Predict"):
        result = predict_churn(input_data)
        st.success(f"### Prediction: **{result}**")

if __name__ == "__main__":
    main()
