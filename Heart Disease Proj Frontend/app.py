import streamlit as st
import pandas as pd
import joblib


model = joblib.load('./1/Heart/KNN_heart.pkl')
scaler = joblib.load('./1/Heart/scaler.pkl')
expected_columns = joblib.load('./1/Heart/heart_columns.pkl')


st.title("Heart Disease Prediction By SURAJ!!")
st.markdown("### Enter the following details to predict whether a person has heart disease or not.")


age = st.slider('Age',1,100,40)
sex = st.select('Sex'['M']['F'])
pain = st.selectbox('Chest Pain Type',['ATA','NAP','ASY','TA'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)',80,200,120)
cholesterol = st.number_input('Serum Cholesterol (mg/dl)',100,600,200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl',[0,1])
resting_ecg = st.number_input('Resting ECG',["Normal","ST","LVH"])
max_hr = st.number_input('Maximum Heart Rate Achieved',60,220,150)
oldPeak  = st.number_input('Oldpeak',0.0,6.0,1.0)
st_slope = st.selectbox('Slope',['Up','Flat','Down'])
exercise_angina = st.selectbox('Exercise Induced Angina',['Y','N'])


if st.button('Predict'):
   raw_ip ={
      'Age':age,
        'Sex':sex,
        'Cholesterol':cholesterol,
        'FastingBS':fasting_bs,
        'RestingBP':resting_bp,
        'ChestPainType_'+pain:1,
        'RestingECG_'+resting_ecg:1,
        'MaxHR':max_hr,
        'Oldpeak':oldPeak,
        'ST_Slope_'+st_slope:1,
        'ExceriseAngina_'+ exercise_angina:1,

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




    