import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load trained models and encoders
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('robust_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Define correct feature order (MUST match training data)
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
cat_features = ['Partner', 'Dependents', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

all_features = cat_features + num_features  # Ensure correct order

# Prediction function
def predict_churn(input_data):
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col in cat_features:
        if col in encoders:
            if input_df[col][0] in encoders[col].classes_:
                input_df[col] = encoders[col].transform([input_df[col][0]])
            else:
                input_df[col] = -1  # Assign unknown categories to -1

    # Convert numerical features
    for col in num_features:
        try:
            input_df[col] = float(input_df[col])  # Convert to float
        except ValueError:
            input_df[col] = 0.0  # Handle invalid input gracefully

    # Reorder columns to match training data
    input_df = input_df[all_features]

    # Scale numerical features correctly
    input_df[num_features] = scaler.transform(input_df[num_features])

    # Make prediction
    prediction = model.predict(input_df)[0]

    return "Yes (Churn)" if prediction == 1 else "No (Not Churn)"

# Streamlit UI
def main():
    st.title("Customer Churn Prediction")
    st.subheader("Enter customer details to predict churn.")

    # User inputs (numerical)
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, step=1)
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, format="%.2f")
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, format="%.2f")
    senior_citizen = st.selectbox('Senior Citizen', [0, 1])

    # User inputs (categorical)
    partner = st.selectbox('Partner', encoders['Partner'].classes_)
    dependents = st.selectbox('Dependents', encoders['Dependents'].classes_)
    multiple_lines = st.selectbox('Multiple Lines', encoders['MultipleLines'].classes_)
    internet_service = st.selectbox('Internet Service', encoders['InternetService'].classes_)
    online_security = st.selectbox('Online Security', encoders['OnlineSecurity'].classes_)
    online_backup = st.selectbox('Online Backup', encoders['OnlineBackup'].classes_)
    device_protection = st.selectbox('Device Protection', encoders['DeviceProtection'].classes_)
    tech_support = st.selectbox('Tech Support', encoders['TechSupport'].classes_)
    streaming_tv = st.selectbox('Streaming TV', encoders['StreamingTV'].classes_)
    streaming_movies = st.selectbox('Streaming Movies', encoders['StreamingMovies'].classes_)
    contract = st.selectbox('Contract', encoders['Contract'].classes_)
    paperless_billing = st.selectbox('Paperless Billing', encoders['PaperlessBilling'].classes_)
    payment_method = st.selectbox('Payment Method', encoders['PaymentMethod'].classes_)

    # Prepare input dictionary
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }

    # Predict on button click
    if st.button('Predict'):
        result = predict_churn(input_data)
        st.success(f"### Prediction: **{result}**")

# Run the app
if __name__ == '__main__':
    main()
