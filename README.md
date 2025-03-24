
# 🧠 Alzheimer's Disease Prediction App (Streamlit + SHAP)

This Streamlit app uses **XGBoost** and **SHAP (SHapley Additive Explanations)** to predict whether a patient may have Alzheimer's Disease based on key health features.

## 🚀 Features

- Predict Alzheimer's using a trained XGBoost classifier
- Explain predictions with SHAP waterfall plots
- Interactive UI powered by Streamlit

## 📁 Included Files

- `app.py` – the Streamlit app
- `alzheimers_disease_data.csv` – sample patient dataset
- `requirements.txt` – dependencies to run the app

## 🧪 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click **New App**, connect your GitHub, and select `app.py`
4. Done 🎉

