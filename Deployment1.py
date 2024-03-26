import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("trained_model.joblib")

# Function to preprocess input data
def preprocess_input(input_data):
    input_array = np.array([[input_data[col] for col in ['Pregnancies', 'Glucose', 'BloodPressure', 
                                                        'SkinThickness', 'Insulin', 'BMI', 
                                                        'DiabetesPedigreeFunction', 'Age']]])
    return input_array

# Function to make predictions
def predict(input_data):
    prediction = model.predict_proba(input_data)[0][1]  # Predict probability of class 1 (diabetes)
    return prediction * 100  # Convert probability to percentage

# Streamlit App
st.title("Diabetes Prediction App")

# Input form
st.sidebar.header("User Input")
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
glucose = st.sidebar.slider("Glucose", 0, 200, 100)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 72)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 20)
insulin = st.sidebar.slider("Insulin", 0, 846, 79)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
age = st.sidebar.slider("Age", 21, 81, 29)

# Make prediction on button click
if st.sidebar.button("Predict"):
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }
    input_array = preprocess_input(input_data)
    prediction = predict(input_array)
    st.success(f"The risk of you getting diabetes is {prediction:.2f}%")
