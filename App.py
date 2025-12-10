# Importing necessary libraries
import pandas as pd
import streamlit as st
import joblib

st.markdown(
    """
    <style>
    .block-container {
    border: 25px solid green;
    outline: 10px solid blue;
    border-radius: 15px;
    padding: 10px 25px;
    }
    p {
    colour: teal;
    }
    h1 {
    font-family: 'Courier New', Courier, monospace;
    font-size: 40px;
    color: navy;
    }
    h2 {
    font-family: 'Arial black', Gadget, sans-serif;
    font-size: 30px;
    color: lightblue;
    }
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Loading trained model
model = joblib.load('diabetes_model.pkl')

# Defining the Streamlit app
st.set_page_config(page_title = "Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction App")
st.write("Melissa Merab - Data Science Project 2")
st.title("Diabetes Risk Assessment Model")
st.write("Fill the details below to assess your likelihood of diabetes.")

# Input fields for user data
with st.form('Prediction Form'):
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input('Blood Pressure Level', min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input('Age', min_value=1, max_value=120, value=30)

    submitted = st.form_submit_button('Predict')

    if submitted:
        # Creating a DataFrame for the input data
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })

        # Making prediction
        prediction = model.predict(input_data)

        # Displaying the result
        if prediction[0] == 1:
            st.error("The model predicts that you are at risk of diabetes.")
        else:
            st.success("The model predicts that you are not at risk of diabetes.")

