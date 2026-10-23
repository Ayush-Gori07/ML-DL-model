import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Breast Cancer Classification", page_icon="🩺", layout="wide")

@st.cache_resource
def load_artifacts():
    model = load_model(BASE_DIR / "model.keras")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    return model, scaler

@st.cache_data
def load_options():
    with open(BASE_DIR / "options.json", encoding="utf-8") as f:
        return json.load(f)

st.title("🩺 Breast Cancer Classification")
st.warning(
    "Educational machine-learning demo only. This model is not a medical device and must not be used "
    "to diagnose, rule out, or make treatment decisions about cancer. Consult a qualified clinician."
)

try:
    model, scaler = load_artifacts()
    opt = load_options()
except (OSError, ValueError) as exc:
    st.error("Model files nahi mile. Pehle `python train_model.py` run karo.")
    st.exception(exc)
    st.stop()

st.write("Cell-measurement features enter karo. Model benign/malignant class ka estimated probability dikhayega.")
values = {}
features = opt["features"]
# Keep inputs manageable by showing them in three columns.
columns = st.columns(3)
for i, feat in enumerate(features):
    lo, hi = opt["ranges"][feat]
    default = opt["means"][feat]
    step = max((hi - lo) / 1000, 0.001)
    with columns[i % 3]:
        values[feat] = st.number_input(
            feat.replace("_", " "), min_value=float(lo), max_value=float(hi),
            value=float(min(max(default, lo), hi)), step=float(step), format="%.4f",
            help="Dataset se li gayi allowed range aur average value."
        )

if st.button("Predict class", type="primary"):
    row = pd.DataFrame([[values[f] for f in features]], columns=features)
    scaled = scaler.transform(row)
    prediction = float(np.asarray(model.predict(scaled, verbose=0)).reshape(-1)[0])
    # The training script maps benign=0 and malignant=1.
    malignant_prob = prediction
    benign_prob = 1.0 - prediction
    if malignant_prob >= 0.5:
        st.error(f"Model output: Malignant class probability {malignant_prob:.1%}")
    else:
        st.success(f"Model output: Benign class probability {benign_prob:.1%}")
    st.caption("These are model probabilities, not a confirmed diagnosis or an individual's clinical risk.")
    st.bar_chart(pd.Series({"Benign class": benign_prob, "Malignant class": malignant_prob}))

try:
    with open(BASE_DIR / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    best = next(r for r in metrics["results"] if r["model"] == metrics["best_model"])
    st.caption(f"Best comparison model: {metrics['best_model']} | Test accuracy: {best['accuracy']:.1%} | ROC-AUC: {best['roc_auc']:.3f}")
except (OSError, KeyError, IndexError, TypeError):
    pass
