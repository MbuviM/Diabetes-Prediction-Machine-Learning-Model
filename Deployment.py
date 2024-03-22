import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load("cnn_model.joblib")

# Function to make predictions
def predict(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict_proba(input_df)[0][1] * 100  # Predicting probability of class 1 (diabetes)
    return prediction

# Streamlit App
st.title("Diabetes Prediction App")

# Input form
with st.sidebar:
    st.header("User Input")
    pregnancies = st.selectbox("Pregnancies", [0, 1])
    glucose = st.slider("Glucose", 0, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 0, 150, 80)
    skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
    insulin = st.slider("Insulin", 0, 900, 100)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    diabetes_pedigree_function = st.slider("Diabetes Pedigree Function", 0.0, 2.0, 0.5)
    age = st.slider("Age", 0, 120, 30)
    gender = st.radio("Gender", ["Male", "Female", "Others"])
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
    smoking_history = st.selectbox("Smoking History", ["Yes", "No"])

# Make prediction on button click
if st.button("Predict"):
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Hypertension": 1 if hypertension == "Yes" else 0,
        "HeartDisease": 1 if heart_disease == "Yes" else 0,
        "SmokingHistory": 1 if smoking_history == "Yes" else 0
    }
    prediction = predict(input_data)
    st.success(f"The risk of you getting diabetes is {prediction:.2f}%")


