import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = load_model("cnn2_model.joblib")

# Function to make predictions
def predict(input_data):
    scaler = StandardScaler()
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.fit_transform(input_df)
    prediction = model.predict(input_scaled)[0][0] * 100  # Predicting probability of class 1 (diabetes)
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
        "Gender": 1 if gender == "Male" else (2 if gender == "Female" else 0),
        "Hypertension": 1 if hypertension == "Yes" else 0,
        "HeartDisease": 1 if heart_disease == "Yes" else 0,
        "SmokingHistory": 1 if smoking_history in ['current', 'former', 'ever', 'not current'] else 0
    }
    prediction = predict(input_data)
    st.success(f"The risk of you getting diabetes is {prediction:.2f}%")



