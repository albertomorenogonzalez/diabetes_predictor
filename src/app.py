import streamlit as st
import pandas as pd
import joblib
clf = joblib.load("model/diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Predictor")

st.title('Diabetes Predictor')

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Number of Pregnancies:')
    glucose = st.number_input('Glucose Level in Blood (mg/dL):')
    blood_pressure = st.number_input('Blood Pressure (mm Hg):')
    skin_thickness = st.number_input('Skin Thickness (mm):')

with col2:
    insulin = st.number_input('Insuline Level in Blood (μU/mL):')
    bmi = st.number_input('Body Mass Index (BMI) (kg/m²):')
    db_pdg_fn = st.number_input('Diabetes Pedigree Function (%):') / 100
    age = st.number_input('Age:')


if st.button('Submit'):
    
    X = pd.DataFrame([
            [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, db_pdg_fn, age]
        ], 
        columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]
        )
    
    prediction = clf.predict(X)[0]
    if prediction:
        st.text("The patient has diabetes.")
    else:
        st.text("The patient does not have diabetes.")

st.divider()

st.write("Author: Alberto Moreno González - ")
st.write("AI and Big Data FP Master at CPIFP Alan Turing (Málaga)")
