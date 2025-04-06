import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model, scaler, and label encoders
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('robust_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Define feature lists
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = ['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']

all_features = num_features + cat_features

# Prediction function
def predict_churn(input_data):
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col in cat_features:
        le = encoders[col]
        if input_df[col][0] in le.classes_:
            input_df[col] = le.transform([input_df[col][0]])
        else:
            input_df[col] = -1  # Handle unknowns

    # Scale numerical features using column names
    num_df = input_df[num_features].astype(float)
    scaled_nums = pd.DataFrame(scaler.transform(num_df), columns=num_features)

    # Combine scaled numerics and encoded categoricals
    final_input = pd.concat([scaled_nums, input_df[cat_features].astype(int)], axis=1)

    # Predict
    prediction = model.predict(final_input)[0]
    return "Likely to Churn" if prediction == 1 else "Not Likely to Churn"

# Streamlit UI
def main():
    st.title("📉 Customer Churn Prediction App")
    st.markdown("Enter customer details to predict if they are likely to churn.")

    # Numerical Inputs
    tenure = st.number_input("Tenure", min_value=0)
    monthly = st.number_input("Monthly Charges", min_value=0.0)
    total = st.number_input("Total Charges", min_value=0.0)

    # Categorical Inputs
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'SeniorCitizen': st.selectbox("Senior Citizen", encoders['SeniorCitizen'].classes_),
        'Partner': st.selectbox("Partner", encoders['Partner'].classes_),
        'Dependents': st.selectbox("Dependents", encoders['Dependents'].classes_),
        'MultipleLines': st.selectbox("Multiple Lines", encoders['MultipleLines'].classes_),
        'InternetService': st.selectbox("Internet Service", encoders['InternetService'].classes_),
        'OnlineSecurity': st.selectbox("Online Security", encoders['OnlineSecurity'].classes_),
        'OnlineBackup': st.selectbox("Online Backup", encoders['OnlineBackup'].classes_),
        'DeviceProtection': st.selectbox("Device Protection", encoders['DeviceProtection'].classes_),
        'TechSupport': st.selectbox("Tech Support", encoders['TechSupport'].classes_),
        'StreamingTV': st.selectbox("Streaming TV", encoders['StreamingTV'].classes_),
        'StreamingMovies': st.selectbox("Streaming Movies", encoders['StreamingMovies'].classes_),
        'Contract': st.selectbox("Contract", encoders['Contract'].classes_),
        'PaperlessBilling': st.selectbox("Paperless Billing", encoders['PaperlessBilling'].classes_),
        'PaymentMethod': st.selectbox("Payment Method", encoders['PaymentMethod'].classes_),
    }

    if st.button("Predict"):
        result = predict_churn(input_data)
        st.success(f"### Prediction: **{result}**")

if __name__ == '__main__':
    main()
