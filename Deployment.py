import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the pre-trained model
model = joblib.load("model.joblib")

# Function to preprocess input data
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])  # Convert input dictionary to DataFrame
    # Map gender to numerical value
    gender_map = {"Male": 1, "Female": 0, "Others": 0}  # Assuming Male=1, Female=0, Others=0
    input_df["gender_Male"] = input_df["gender"].map(gender_map)
    input_df["gender_Other"] = (input_df["gender"] == "Others").astype(int)
    
    # Map smoking history to numerical value
    smoking_history_map = {"never": 1, "No Info": 1, "current": 1, "former": 1, "ever": 1, "not current": 1}  # Assuming all are 1
    input_df["smoking_history_current"] = input_df["smoking_history"].map(smoking_history_map)
    input_df["smoking_history_ever"] = 1  # Assuming all are 1
    input_df["smoking_history_former"] = 1  # Assuming all are 1
    input_df["smoking_history_never"] = 1  # Assuming all are 1
    input_df["smoking_history_not current"] = 1  # Assuming all are 1
    
    # Drop unused columns
    input_df.drop(columns=["gender", "smoking_history"], inplace=True)
    
    return input_df

# Function to make predictions
def predict(input_data):
    input_data_processed = preprocess_input(input_data)
    input_data_scaled = scaler.transform(input_data_processed)  # Assuming 'scaler' is defined
    prediction = model.predict(input_data_scaled)[0] * 100  # Predicting probability of class 1 (diabetes)
    return prediction

# Streamlit App
st.title("Diabetes Prediction App")

# Input form
st.sidebar.header("User Input")
gender = st.sidebar.radio("Gender", ["Male", "Female", "Others"])
age = st.sidebar.slider("Age", 0, 120, 30)
hypertension = st.sidebar.selectbox("Hypertension", ["Yes", "No"])
heart_disease = st.sidebar.selectbox("Heart Disease", ["Yes", "No"])
smoking_history = st.sidebar.selectbox("Smoking History", ['never', 'No Info', 'current', 'former', 'ever', 'not current'])
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
HbA1c_level = st.sidebar.slider("HbA1c Level", 0.0, 15.0, 6.0)
blood_glucose_level = st.sidebar.slider("Blood Glucose Level", 0, 500, 100)

# Make prediction on button click
if st.sidebar.button("Predict"):
    input_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }
    prediction = predict(input_data)
    st.success(f"The risk of you getting diabetes is {prediction:.2f}%")











