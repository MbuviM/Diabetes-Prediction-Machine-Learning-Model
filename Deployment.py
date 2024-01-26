import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load("trained_model.joblib")

# Function to make predictions
def predict(input_data):
    input_df = pd.DataFrame([input_data])
    predictions = model.predict(input_df)
    return predictions[0]

# Streamlit App
st.title("Diabetes Prediction App")

# Input form
st.sidebar.header("User Input")
pregnancies = st.sidebar.number_input("Pregnancies", 0, 17, 1)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 99, 20)
diabetes_pedigree_function = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.0, 0.5)

# Make prediction on button click
if st.sidebar.button("Predict"):
    input_data = {
        "Pregnancies": pregnancies,
        "SkinThickness": skin_thickness,
        "DiabetesPedigreeFunction": diabetes_pedigree_function
    }
    prediction = predict(input_data)
    st.success(f"The model predicts: {prediction}")

