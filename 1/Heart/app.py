import streamlit as st
import pandas as pd
import joblib

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model = joblib.load(os.path.join(BASE_DIR, 'KNN_heart.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
expected_columns = joblib.load(os.path.join(BASE_DIR, 'columns.pkl'))

st.title("Heart Disease Prediction By SURAJ!!")
st.markdown("### Enter the following details to predict whether a person has heart disease or not.")

age = st.slider('Age', 1, 100, 40)
sex = st.selectbox('Sex', ['M', 'F'])
pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
cholesterol = st.number_input('Serum Cholesterol (mg/dl)', 100, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input('Maximum Heart Rate Achieved', 60, 220, 150)
oldPeak = st.number_input('Oldpeak', 0.0, 6.0, 1.0)
st_slope = st.selectbox('Slope', ['Up', 'Flat', 'Down'])
exercise_angina = st.selectbox('Exercise Induced Angina', ['Y', 'N'])

if st.button('Predict'):
    raw_ip = {
        'Age': age,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingBP': resting_bp,
        'MaxHR': max_hr,
        'Oldpeak': oldPeak,

        'Sex_M': 1 if sex == 'M' else 0,
        'Sex_F': 1 if sex == 'F' else 0,

        'ChestPainType_ATA': 1 if pain == 'ATA' else 0,
        'ChestPainType_NAP': 1 if pain == 'NAP' else 0,
        'ChestPainType_ASY': 1 if pain == 'ASY' else 0,
        'ChestPainType_TA': 1 if pain == 'TA' else 0,

        'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
        'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
        'RestingECG_LVH': 1 if resting_ecg == 'LVH' else 0,

        'ST_Slope_Up': 1 if st_slope == 'Up' else 0,
        'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
        'ST_Slope_Down': 1 if st_slope == 'Down' else 0,

        'ExerciseAngina_Y': 1 if exercise_angina == 'Y' else 0,
        'ExerciseAngina_N': 1 if exercise_angina == 'N' else 0,
    }

   
    input_df = pd.DataFrame([raw_ip])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_ip = scaler.transform(input_df)

    prediction = model.predict(scaled_ip)[0]

    if prediction == 1:
        st.error("The person is likely to have Heart Disease.")
    else:
        st.success("The person is unlikely to have Heart Disease.")
