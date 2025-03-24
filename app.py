
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("alzheimers_disease_data.csv")
    df = df.drop(columns=["PatientID", "DoctorInCharge"])
    return df

df = load_data()
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

explainer = shap.Explainer(model)

st.title("🧠 Alzheimer's Disease Prediction (XGBoost + SHAP)")
st.markdown("Enter patient data to predict Alzheimer’s and visualize feature impact.")

user_input = {}
with st.form("prediction_form"):
    for feature in X.columns:
        val = st.number_input(f"{feature}", value=float(X[feature].mean()))
        user_input[feature] = val
    submitted = st.form_submit_button("Predict & Explain")

if submitted:
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    result = "🟥 Alzheimer's Positive" if prediction == 1 else "🟩 No Alzheimer's"
    st.subheader(f"Prediction Result: {result}")

    shap_input = explainer(input_df)
    st.subheader("🔍 SHAP Feature Impact")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_input[0], max_display=10, show=False)
    st.pyplot(fig)
