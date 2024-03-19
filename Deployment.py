import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load("trained_model.joblib")

# Function to make predictions
def predict(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict_proba(input_df)[0][1] * 100  # Predicting probability of class 1 (diabetes)
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
        "Age": age
    }
    prediction = predict(input_data)
    st.success(f"The risk of you getting diabetes is {prediction:.2f}%")
