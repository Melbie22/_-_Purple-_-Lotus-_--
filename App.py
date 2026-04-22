# Importing necessary libraries
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader 
import pandas as pd
import numpy as np
import joblib

# ============================================
#          PAGE CONFIGURATION 
# ============================================

st.set_page_config(
    page_title="Diabetes Risk Assessment Tool",
    page_icon="🏥",
    layout="wide"
)

# ============================================
#          LOAD EXTERNAL CSS FILE
# ============================================

# Function to load CSS from external file
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styling.")

# Call the function to load your CSS
load_css('style.css')

# =======================
#  LOGIN AUTHENTICATION
# =======================

# Load the configuration file containing user credentials
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Configuration file 'config.yaml' not found. Please ensure it exists in the same directory as this app.")
    st.stop()

# Create the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Display the login form 
authenticator.login(location='main')

# Check login status from session state
if st.session_state["authentication_status"] is False:
    st.error('Username or password is incorrect. Please try again.')
    st.stop()

if st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password to access the Diabetes Risk Assessment Tool.')
    st.stop()

# If we reach here, login was successful
if st.session_state["authentication_status"]:
    with st.sidebar:
        authenticator.logout(location='sidebar')
        st.write(f'**Welcome, {st.session_state["name"]}!**')
        st.markdown("---")

# Loading trained model
model = joblib.load('diabetes_model.pkl')
#scaler = joblib.load('scaler.pkl')

#   VALIDATION FUNCTION

def validate_biological_inputs(input_data):
    """Check if all inputs are within biological ranges"""
    errors = []
    
    # Debug: Show what was received
    #st.write("Debug - input_data received:", input_data)
    #st.write("Debug - types:", [type(f) for f in input_data])
    
    # Define validation rules: (name, min_value, max_value)
    validation_rules = [
        ("Pregnancies", 0, 20),
        ("Glucose", 40, 300),
        ("Blood Pressure", 40, 200),
        ("Skin Thickness", 0, 100),
        ("Insulin", 0, 1000),
        ("BMI", 10, 60),
        ("Diabetes Pedigree Function", 0, 3),
        ("Age", 0, 120)
    ]
    
    # Validate each feature
    for i, (name, min_val, max_val) in enumerate(validation_rules):
        value = input_data[i]
        
        # Convert tuple to number if needed (safety check)
        if isinstance(value, tuple):
            value = value[0] if len(value) > 0 else 0
        
        if value < min_val or value > max_val:
            errors.append(f"❌ {name} ({value}) should be between {min_val} and {max_val}")
    
    return errors
# Defining the Streamlit app
st.set_page_config(page_title = "Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction App")
st.write("MM Data Science Project")
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
    # Create a simple list for the input data
        input_data = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]
        
        # Validate inputs
        errors = validate_biological_inputs(input_data)
        
        # Check for validation errors
        if errors:
            # Show errors if any
            for error in errors:
                st.error(error)
            st.warning("⚠️ Please correct the invalid values before predicting")
        else:
            # Convert to 2D array for prediction
            input_array = np.array([input_data])
            
            # Make prediction
            prediction = model.predict(input_array)
            
            # TEMPORARY DEBUG - SEE WHAT PREDICTION VALUE IS
           # st.write(f"**Debug - Raw prediction value:** {prediction}")
            #st.write(f"**Debug - Prediction type:** {type(prediction)}")
            #st.write(f"**Debug - Prediction[0]:** {prediction[0]}")
            
            # Display the result
            if prediction[0] == 1:
                st.error("The model predicts that you are at high risk of diabetes. "
                        "Please consult a healthcare professional for further evaluation.")
            else:
                st.success("The model predicts that you are at low risk of diabetes. "
                        "Maintain a healthy lifestyle.")

# Display disclaimer (OUTSIDE the if submitted block)
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This tool is for educational and screening purposes only. "
           "It does not replace professional medical advice, diagnosis, or treatment. "
           "Always consult a qualified healthcare provider for medical decisions.")
    
