import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    try:
        model = load_model(BASE_DIR / "model.keras")
    except Exception as exc:
        st.error(f"model.keras load nahi ho paaya: {exc}")
        st.stop()
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    return model, scaler


@st.cache_data
def load_options():
    with open(BASE_DIR / "options.json") as f:
        return json.load(f)


model, scaler = load_artifacts()
opt = load_options()

st.title("🌸 Iris Flower Classifier")
st.write("Flower ki measurements daalo, neural network batayega ki woh kaunsi **Iris species** hai.")

values = {}
col1, col2 = st.columns(2)
for i, feat in enumerate(opt["features"]):
    lo, hi = opt["ranges"][feat]
    with (col1 if i % 2 == 0 else col2):
        values[feat] = st.slider(feat, lo, hi, opt["means"][feat], 0.1)

if st.button("Predict species"):
    row = pd.DataFrame([values], columns=opt["features"])
    probs = model.predict(scaler.transform(row), verbose=0)[0]
    pred = opt["classes"][int(np.argmax(probs))]
    st.success(f"Predicted species: **Iris-{pred}** ({probs.max():.1%} confidence)")
    st.bar_chart(pd.Series(probs, index=opt["classes"]))

try:
    with open("metrics.json") as f:
        m = json.load(f)
    best = next(r for r in m["results"] if r["model"] == m["best_model"])
    st.caption(f"Model: {m['best_model']} | Test accuracy: {best['accuracy']:.1%}")
except (FileNotFoundError, StopIteration):
    pass
