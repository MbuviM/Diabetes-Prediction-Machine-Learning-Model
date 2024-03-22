import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the pre-trained model
model = load_model("cnn2_model.h5")

# Function to preprocess input data
def preprocess_input(input_data):
    # Map gender to numerical value
    gender_map = {"Male": 1, "Female": 2, "Others": 0}
    input_data["Gender"] = gender_map[input_data["Gender"]]
    
    # Map categorical variables to numerical values
    binary_map = {"Yes": 1, "No": 0}
    input_data["Hypertension"] = binary_map[input_data["Hypertension"]]
    input_data["HeartDisease"] = binary_map[input_data["HeartDisease"]]
    
    # Map smoking history to binary values
    smoking_map = {"never": 0, "No Info": 0, "current": 1, "former": 1, "ever": 1, "not current": 1}
    input_data["SmokingHistory"] = smoking_map[input_data["SmokingHistory"]]
    
    return np.array([[input_data[col] for col in ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                                                  "Gender", "Hypertension", "HeartDisease", "SmokingHistory"]]])

# Function to make predictions
def predict(input_data):
    input_data_scaled = preprocess_input(input_data)
    prediction = model.predict(input_data_scaled)[0][0] * 100  # Predicting probability of class 1 (diabetes)
    return prediction

# Streamlit App
st.title("Diabetes Prediction App")

# Input form
st.sidebar.header("User Input")
pregnancies = st.sidebar.selectbox("Pregnancies", [0, 1])
glucose = st.sidebar.slider("Glucose", 0, 200, 100)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 150, 80)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 100)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
diabetes_pedigree_function = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.0, 0.5)
age = st.sidebar.slider("Age", 0, 120, 30)
gender = st.sidebar.radio("Gender", ["Male", "Female", "Others"])
hypertension = st.sidebar.selectbox("Hypertension", ["Yes", "No"])
heart_disease = st.sidebar.selectbox("Heart Disease", ["Yes", "No"])
smoking_history = st.sidebar.selectbox("Smoking History", ['never', 'No Info', 'current', 'former', 'ever', 'not current'])

# Make prediction on button click
if st.sidebar.button("Predict"):
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age,
        "Gender": gender,
        "Hypertension": hypertension,
        "HeartDisease": heart_disease,
        "SmokingHistory": smoking_history
    }
    prediction = predict(input_data)
    st.success(f"The risk of you getting diabetes is {prediction:.2f}%")
