# Importing necessary libraries
from pyexpat import features
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

 # Validate inputs
        
def validate_biological_inputs(features):
    """Check if all inputs are within biological ranges"""
    errors = []
    
    validation_rules = {
        'Pregnancies': (0, 20, features[0]),
        'Glucose': (40, 300, features[1]),
        'Blood Pressure': (40, 200, features[2]),
        'Skin Thickness': (0, 100, features[3]),
        'Insulin': (0, 1000, features[4]),
        'BMI': (10, 60, features[5]),
        'Diabetes Pedigree Function': (0, 3, features[6]),
        'Age': (0, 120, features[7])
    }
    
    for name, (min_val, max_val, value) in validation_rules.items():
        if value < min_val or value > max_val:
            errors.append(f"❌ {name} ({value}) should be between {min_val} and {max_val}")
    
    return errors

# Defining the Streamlit app
st.set_page_config(page_title = "Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction App")
st.write("M_M Data Science Project")
st.title("Diabetes Risk Assessment Model")
st.write("Fill the details below to assess your likelihood of diabetes.")

# Input fields for user data
with st.form('Prediction Form'):
    pregnancies = st.number_input('Number of Pregnancies', min_value = 0, max_value = 20, value = 0)
    glucose = st.number_input('Glucose Level', min_value = 0, max_value = 200, value = 100)
    blood_pressure = st.number_input('Blood Pressure Level', min_value = 0, max_value = 150, value = 70)
    skin_thickness = st.number_input('Skin Thickness', min_value = 0, max_value = 100, value = 20)
    insulin = st.number_input('Insulin Level', min_value = 0, max_value = 900, value = 80)
    bmi = st.number_input('BMI', min_value = 0.0, max_value = 70.0, value = 25.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value = 0.0, max_value = 2.5, value = 0.5)
    age = st.number_input('Age', min_value = 1, max_value = 120, value = 30)

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

        errors = validate_biological_inputs(features)
        
        if errors:
            # Show errors if any
            for error in errors:
                st.error(error)
            st.warning("⚠️ Please correct the invalid values before predicting")
        else:

            # Making prediction
            prediction = model.predict(input_data)
    
            # Displaying the result
            if prediction[0] == 1:
                st.error("The model predicts that you are at high risk of diabetes. Please consult a healthcare professional for further evaluation.")
            else:
                st.success("The model predicts that you are at low risk of diabetes. Maintain a healthy lifestyle.")
