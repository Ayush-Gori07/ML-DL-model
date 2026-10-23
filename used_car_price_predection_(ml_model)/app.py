import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_options():
    with open("options.json") as f:
        return json.load(f)


model = load_model()
opt = load_options()

st.title("🚗 Used Car Price Predictor")
st.write("Car ki details daalo, model uski estimated price batayega (₹ mein).")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", sorted(opt["brand_models"].keys()))
    car_model = st.selectbox("Model", opt["brand_models"][brand])
    fuel = st.selectbox("Fuel type", opt["fuel"])
with col2:
    transmission = st.selectbox("Transmission", opt["transmission"])
    owner = st.selectbox("Owner", opt["owner"])
    age = st.number_input("Age (years)", 0, opt["max_age"], 5)

km = st.number_input("Kilometers driven", 0, int(opt["max_km"] * 1000), 40000, 1000)

if st.button("Predict price"):
    row = pd.DataFrame(
        [[brand, car_model, fuel, transmission, owner, age, km / 1000]],
        columns=["Brand", "model", "FuelType", "Transmission", "Owner", "Age", "kmDriven"],
    )
    # model log(price in thousands) predict karta hai
    price = float(np.expm1(model.predict(row)[0])) * 1000
    st.success(f"Estimated price: ₹ {price:,.0f}  (≈ {price / 100000:.2f} lakh)")
    st.caption("Yeh sirf ek estimate hai; asli price condition aur city pe bhi depend karti hai.")

try:
    with open("metrics.json") as f:
        m = json.load(f)
    best = next(r for r in m["results"] if r["model"] == m["best_model"])
    st.caption(f"Model: {m['best_model']} | Typical error (MedAE): ₹ {best['MedAE'] * 1000:,.0f}")
except (FileNotFoundError, StopIteration):
    pass
