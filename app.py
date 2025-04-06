import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load trained model, scaler, and encoders
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

all_features = cat_features + num_features  # Order matters

def predict_churn(input_data):
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col in cat_features:
        le = encoders[col]
        val = input_df[col][0]
        if val in le.classes_:
            input_df[col] = le.transform([val])
        else:
            input_df[col] = -1  # Handle unseen category

    input_df[cat_features] = input_df[cat_features].astype('object')

    # Scale numerical features
    num_df = input_df[num_features].astype(float)
    scaled_nums = pd.DataFrame(scaler.transform(num_df), columns=num_features)

    # Combine all features
    final_df = pd.concat([input_df[cat_features], scaled_nums], axis=1)
    final_df = final_df[all_features]

    prediction = model.predict(final_df)[0]
    return "Likely to Churn" if prediction == 1 else "Not Likely to Churn"

def main():
    st.title("📉 Customer Churn Prediction App")
    st.markdown("Enter customer details to predict if they are likely to churn.")

    # Numerical inputs
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, step=1.0)
    total = st.number_input("Total Charges ($)", min_value=0.0, step=1.0)

    # Categorical inputs
    input_data = {
        'SeniorCitizen': st.selectbox('Senior Citizen', encoders['SeniorCitizen'].classes_),
        'Partner': st.selectbox('Partner', encoders['Partner'].classes_),
        'Dependents': st.selectbox('Dependents', encoders['Dependents'].classes_),
        'MultipleLines': st.selectbox('Multiple Lines', encoders['MultipleLines'].classes_),
        'InternetService': st.selectbox('Internet Service', encoders['InternetService'].classes_),
        'OnlineSecurity': st.selectbox('Online Security', encoders['OnlineSecurity'].classes_),
        'OnlineBackup': st.selectbox('Online Backup', encoders['OnlineBackup'].classes_),
        'DeviceProtection': st.selectbox('Device Protection', encoders['DeviceProtection'].classes_),
        'TechSupport': st.selectbox('Tech Support', encoders['TechSupport'].classes_),
        'StreamingTV': st.selectbox('Streaming TV', encoders['StreamingTV'].classes_),
        'StreamingMovies': st.selectbox('Streaming Movies', encoders['StreamingMovies'].classes_),
        'Contract': st.selectbox('Contract', encoders['Contract'].classes_),
        'PaperlessBilling': st.selectbox('Paperless Billing', encoders['PaperlessBilling'].classes_),
        'PaymentMethod': st.selectbox('Payment Method', encoders['PaymentMethod'].classes_),
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    if st.button("Predict"):
        result = predict_churn(input_data)
        st.success(f"### Prediction: **{result}**")

if __name__ == "__main__":
    main()
