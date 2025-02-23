import streamlit as st
import numpy as np
import joblib


from train import accuracy

model = joblib.load('model/model.pkl')
st.title('Diabetic Prediction Model')
pationt_name = st.text_input('pationt_name')
Pregnancies = st.text_input("Enter No Pregnancies: ")
Glucose = st.text_input("Enter Glucose: ")
BloodPressure = st.text_input("Enter Blood Pressure: ")
SkinThickness = st.text_input("Enter Skin Thickness: ")
Insulin = st.text_input("Enter Insulin: ")
BMI = st.text_input("Enter BMI: ")
DiabetesPedigreeFunction = st.text_input("Enter Diabetes Pedigree Function: ")
Age = st.text_input("Enter Age: ")

if st.button('Predict'):
    if all([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]):
        input_data = np.array([
            int(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
            float(Insulin), float(BMI), float(DiabetesPedigreeFunction), int(Age)
        ]).reshape(1,-1)

        prediction = model.predict(input_data)
        if prediction == 0:
            result = "No Diabetes"
        else:
            result = "Diabetes"

        st.success(f"The patient {pationt_name} is diagnosed with: {result}. The Prediction Accuracy is:")



    else:
        st.error("Please fill all the required fields.")