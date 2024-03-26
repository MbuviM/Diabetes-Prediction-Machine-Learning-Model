import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the pre-trained model
model = joblib.load("model.joblib")

# Function to preprocess input data
def preprocess_input(input_data):
    # Map gender to numerical value
    gender_map = {"Male": 1, "Female": 2, "Others": 0}
    input_data["gender"] = gender_map[input_data["gender"]]
    
    # Map hypertension to numerical value
    hypertension_map = {"Yes": 1, "No": 0}
    input_data["hypertension"] = hypertension_map[input_data["hypertension"]]
    
    # Map heart disease to numerical value
    heart_disease_map = {"Yes": 1, "No": 0}
    input_data["heart_disease"] = heart_disease_map[input_data["heart_disease"]]
    
    # Map smoking history to numerical value
    smoking_history_map = {"never": 0, "No Info": 0, "current": 1, "former": 1, "ever": 1, "not current": 1}
    input_data["smoking_history"] = smoking_history_map[input_data["smoking_history"]]
    
    # Encode categorical variables
    encoded_gender = [0, 0]  # Initialize encoded gender features
    if input_data["gender"] == 1:  # Male
        encoded_gender[0] = 1
    elif input_data["gender"] == 2:  # Female
        encoded_gender[1] = 1
    
    encoded_smoking_history = [0, 0, 0, 0, 0]  # Initialize encoded smoking history features
    if input_data["smoking_history"] == "current":
        encoded_smoking_history[0] = 1
    elif input_data["smoking_history"] == "former":
        encoded_smoking_history[1] = 1
    elif input_data["smoking_history"] == "ever":
        encoded_smoking_history[2] = 1
    elif input_data["smoking_history"] == "not current":
        encoded_smoking_history[3] = 1
    
    return np.array([[input_data["gender"], input_data["age"], input_data["hypertension"], input_data["heart_disease"],
                      input_data["smoking_history"], input_data["bmi"], input_data["HbA1c_level"],
                      input_data["blood_glucose_level"]] + encoded_gender + encoded_smoking_history])

# Function to make predictions
def predict(input_data):
    input_data_scaled = preprocess_input(input_data)
    print("Number of features in input data:", input_data_scaled.shape[1])  # Debugging statement
    prediction = model.predict(input_data_scaled)[0][0] * 100  # Predicting probability of class 1 (diabetes)
    return prediction

# Streamlit App
st.title("Diabetes Prediction App")

# Input form
st.sidebar.header("User Input")
gender = st.sidebar.radio("Gender", ["Male", "Female", "Others"])
age = st.sidebar.slider("Age", 0, 120, 30)
hypertension = st.sidebar.radio("Hypertension", ["Yes", "No"])
heart_disease = st.sidebar.radio("Heart Disease", ["Yes", "No"])
smoking_history = st.sidebar.radio("Smoking History", ['never', 'No Info', 'current', 'former', 'ever', 'not current'])
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







