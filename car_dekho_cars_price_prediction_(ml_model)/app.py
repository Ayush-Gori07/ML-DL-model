import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CarDekho Price Predictor", page_icon="🚙")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_options():
    with open("options.json") as f:
        return json.load(f)


model = load_model()
opt = load_options()

st.title("🚙 Used Car Selling Price Predictor")
st.write("Car ki details daalo, model uski estimated selling price batayega (₹ mein).")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", sorted(opt["brand_models"].keys()))
    car_model = st.selectbox("Model", opt["brand_models"][brand])
    fuel = st.selectbox("Fuel type", opt["fuel"])
    year = st.number_input("Manufacturing year", opt["min_year"], opt["max_year"], opt["max_year"] - 5)
with col2:
    seller_type = st.selectbox("Seller type", opt["seller_type"])
    transmission = st.selectbox("Transmission", opt["transmission"])
    owner = st.selectbox("Owner", opt["owner"])
    km = st.number_input("Kilometers driven", 0, opt["max_km"], min(40000, opt["max_km"]), 1000)

if st.button("Predict price"):
    row = pd.DataFrame(
        [[fuel, seller_type, transmission, owner, brand, car_model, year, km]],
        columns=["fuel", "seller_type", "transmission", "owner", "brand", "model", "year", "km_driven"],
    )
    price = float(np.expm1(model.predict(row)[0]))
    st.success(f"Estimated price: ₹ {price:,.0f}  (≈ {price / 100000:.2f} lakh)")
    st.caption("Yeh sirf ek estimate hai; asli price condition aur city pe bhi depend karti hai.")

try:
    with open("metrics.json") as f:
        m = json.load(f)
    best = next(r for r in m["results"] if r["model"] == m["best_model"])
    st.caption(f"Model: {m['best_model']} | Typical error (MedAE): ₹ {best['MedAE']:,.0f}")
except (FileNotFoundError, StopIteration):
    pass
