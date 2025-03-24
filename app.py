import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Alzheimer's Prediction", layout="centered")

@st.cache_data
def load_data():
    df = pd.read_csv("alzheimers_disease_data.csv")
    df = df.drop(columns=["PatientID", "DoctorInCharge"], errors="ignore")
    return df

df = load_data()

if df.shape[0] < 5:
    st.warning("⚠️ The dataset has too few records. Please upload a dataset with at least 5 rows.")
    st.stop()

# Prepare features and target
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# UI
st.title("🧠 Alzheimer's Disease Prediction (XGBoost + SHAP)")
st.markdown("Enter patient information below to predict Alzheimer's and understand the feature impact using SHAP.")

user_input = {}
for feature in X.columns:
    default_val = float(X[feature].mean())
    user_input[feature] = st.number_input(f"{feature}", value=default_val)

if st.button("Predict & Explain"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    result = "✅ No Alzheimer's" if prediction == 0 else "⚠️ Alzheimer's Positive"
    st.subheader(f"Prediction: {result}")

    shap_input = explainer(input_df)
    st.subheader("🔍 Feature Contribution (SHAP)")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_input[0], max_display=10, show=False)
    st.pyplot(fig)
