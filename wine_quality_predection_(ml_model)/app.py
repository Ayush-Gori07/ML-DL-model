import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🍷 Red Wine Quality Predictor")
st.write("Wine ke chemical properties daalo, model batayega ki wine **good (quality ≥ 7)** hai ya nahi.")

col1, col2 = st.columns(2)
with col1:
    volatile_acidity = st.number_input("Volatile acidity", 0.10, 1.60, 0.52, 0.01)
    citric_acid = st.number_input("Citric acid", 0.00, 1.00, 0.26, 0.01)
    total_so2 = st.number_input("Total sulfur dioxide", 5.0, 300.0, 38.0, 1.0)
with col2:
    density = st.number_input("Density", 0.9900, 1.0050, 0.9968, 0.0001, format="%.4f")
    sulphates = st.number_input("Sulphates", 0.30, 2.00, 0.62, 0.01)
    alcohol = st.number_input("Alcohol (%)", 8.0, 15.0, 10.2, 0.1)

if st.button("Predict"):
    row = pd.DataFrame(
        [[volatile_acidity, citric_acid, total_so2, density, sulphates, alcohol]],
        columns=[
            "volatile acidity",
            "citric acid",
            "total sulfur dioxide",
            "density",
            "sulphates",
            "alcohol",
        ],
    )
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    if pred == 1:
        st.success(f"Good quality wine 🎉  (probability: {prob:.0%})")
    else:
        st.warning(f"Not a high-quality wine  (good-wine probability: {prob:.0%})")

try:
    with open("metrics.json") as f:
        m = json.load(f)
    st.caption(
        f"Model: Random Forest | Test accuracy: {m['accuracy']:.1%} | "
        f"Good-wine recall: {m['good_recall']:.1%}"
    )
except FileNotFoundError:
    pass
