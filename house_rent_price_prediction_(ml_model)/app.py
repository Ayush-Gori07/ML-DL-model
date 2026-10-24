import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Rent Predictor", page_icon="🏠")

OTHER = "Other (list mein nahi hai)"
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "model.pkl")


@st.cache_data
def load_options():
    with open(BASE_DIR / "options.json") as f:
        return json.load(f)


model = load_model()
opt = load_options()

st.title("🏠 House Rent Predictor")
st.write("Ghar ki details daalo, model uska estimated **monthly rent** (₹) batayega.")

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", sorted(opt["city_localities"].keys()))
    locality = st.selectbox("Area / Locality", opt["city_localities"][city] + [OTHER])
    area_type = st.selectbox("Area type", opt["area_type"])
    furnishing = st.selectbox("Furnishing", opt["furnishing"])
    contact = st.selectbox("Point of contact", opt["contact"])
with col2:
    bhk = st.number_input("BHK", 1, opt["max_bhk"], 2)
    size = st.number_input("Size (sq ft)", opt["min_size"], opt["max_size"], min(800, opt["max_size"]), 50)
    bathroom = st.number_input("Bathrooms", 1, 4, 2)
    tenant = st.selectbox("Tenant preferred", opt["tenant"])
    floor = st.number_input("Floor number (Ground = 0)", 0, 60, 1)
    total_floor = st.number_input("Total floors in building", 1, 80, 4)

if st.button("Predict rent"):
    row = pd.DataFrame(
        [
            [
                area_type,
                city,
                furnishing,
                contact,
                "unknown_locality" if locality == OTHER else locality,
                bhk,
                size,
                bathroom,
                int("family" in tenant.lower()),
                int("bachelor" in tenant.lower()),
                floor,
                total_floor,
            ]
        ],
        columns=[
            "Area Type", "City", "Furnishing Status", "Point of Contact", "Area Locality",
            "BHK", "Size", "Bathroom", "for_family", "for_bachelor", "my_floor", "total_floor",
        ],
    )
    rent = float(np.expm1(model.predict(row)[0]))
    st.success(f"Estimated rent: ₹ {rent:,.0f} / month")
    st.caption(
        f"Model sirf ₹{opt['min_rent']:,} – ₹{opt['max_rent']:,} ke beech ke rents pe train hua hai, "
        "isliye bahut mehngi properties ka estimate bharosemand nahi hoga."
    )

try:
    with open("metrics.json") as f:
        m = json.load(f)
    best = next(r for r in m["results"] if r["model"] == m["best_model"])
    st.caption(f"Model: {m['best_model']} | Typical error (MedAE): ₹ {best['MedAE']:,.0f}")
except (FileNotFoundError, StopIteration):
    pass
