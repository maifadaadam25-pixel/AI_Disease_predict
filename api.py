

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# =========================
# 1. LOAD MODEL + ENCODER
# =========================
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label__encoder.pkl")

# =========================
# 2. PAGE CONFIG
# =========================
st.set_page_config(
    page_title="DIABETES DISEASE Prediction AI",
    layout="centered"
)

st.title("🧠 DIABETES Prediction System")
st.write("Enter symptom levels below to predict health state.")

# =========================
# 3. FEATURE INPUTS
# =========================
st.header("Patient Symptom Inputs")

Age=st.number_input("Age", 0, 100)
Gender=st.selectbox("Gender", ["Male", "Female"])
if Gender=="Male":
    Gender=1
else:
    Gender=0
BMI=st.number_input("BMI", 0, 100)
Blood_Pressure=st.slider("Blood Pressure",min_value=0, max_value=10,value=5)
Cholestrol=st.slider("Cholestrol", 0, 10,5)
Glucose=st.slider("Glucose", 0, 10,5)
Smoking=st.slider("Smoking", 0, 10,5)
Alcohol_Consumption=st.slider("Alcohol Consumption", 0, 10,5)
Exercise=st.slider("Exercise", 0, 10,5)

# =========================
# 4. INPUT VECTOR
# =========================
input_data = np.array([[
    Age, Gender, BMI, Blood_Pressure, Cholestrol, Glucose, Smoking,Alcohol_Consumption, Exercise
]])

# =========================
# 5. PREDICTION
# =========================
if st.button("PREDICT DIABETES STATE"):

    prediction = model.predict(input_data)
    prediction_label = label_encoder.inverse_transform(prediction)[0]

    st.subheader("🧾 Predicted Result")


    if prediction_label == 0:
        st.success("NO DIABETES")
    elif prediction_label== 1:
        st.success("DIABETES DETECTED")


    # =========================
    # 6. PROBABILITY DISPLAY
    # =========================
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]

        prob_df = pd.DataFrame({
            "Class": label_encoder.classes_,
            "Probability": probs
        })

        st.subheader("📊 Prediction Probabilities")
        st.bar_chart(prob_df.set_index("Class"))