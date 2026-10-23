"""Train a baseline classifier and a Keras neural network for breast-cancer dataset.

Run from this folder: python train_model.py
Dataset: data.csv or data/data.csv (Wisconsin Diagnostic Breast Cancer CSV).
If no CSV is found, scikit-learn's bundled Wisconsin dataset is used.
Outputs: model.keras, scaler.pkl, options.json, metrics.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
CANDIDATES = [BASE_DIR / "data.csv", BASE_DIR / "data" / "data.csv"]


def load_dataset():
    csv_path = next((p for p in CANDIDATES if p.exists()), None)
    if csv_path:
        df = pd.read_csv(csv_path)
        df = df.drop(columns=[c for c in df.columns if str(c).lower().startswith("unnamed")], errors="ignore")
        df = df.drop(columns=[c for c in df.columns if str(c).lower() in {"id", "id_number", "id number"}], errors="ignore")
        diagnosis_col = next((c for c in df.columns if c.lower() in {"diagnosis", "target", "label"}), None)
        if diagnosis_col is None:
            raise ValueError("CSV mein diagnosis/target/label column nahi mila.")
        labels = df.pop(diagnosis_col)
        labels = labels.astype(str).str.strip().str.upper()
        mapping = {"B": 0, "BENIGN": 0, "0": 0, "M": 1, "MALIGNANT": 1, "1": 1}
        y = labels.map(mapping)
        if y.isna().any():
            unknown = sorted(labels[y.isna()].unique())
            raise ValueError(f"Unknown diagnosis labels: {unknown}")
        y = y.astype(int)
        X = df.apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all")
        valid = X.notna().all(axis=1) & y.notna()
        X, y = X.loc[valid], y.loc[valid].astype(int)
        if X.empty or X.shape[1] == 0:
            raise ValueError("CSV mein usable numeric feature columns nahi mile.")
        print(f"Loaded dataset: {csv_path}")
        return X, y

    print("No data.csv found; using scikit-learn's bundled Wisconsin breast-cancer dataset.")
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(1 - dataset.target, name="diagnosis")
    return X, y


X, y = load_dataset()
if y.nunique() != 2:
    raise ValueError("Binary classification ke liye exactly 2 classes chahiye.")
FEATURES = list(X.columns)
print(f"Rows: {len(X)} | Features: {len(FEATURES)} | Label convention: 0=Benign, 1=Malignant")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Fit preprocessing only on training data to avoid test-data leakage.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Baseline model.
baseline = LogisticRegression(max_iter=3000, random_state=42)
baseline.fit(X_train_s, y_train)
baseline_probs = baseline.predict_proba(X_test_s)[:, 1]
baseline_preds = (baseline_probs >= 0.5).astype(int)
baseline_acc = accuracy_score(y_test, baseline_preds)
baseline_auc = roc_auc_score(y_test, baseline_probs)
print(f"\nLogistic Regression accuracy: {baseline_acc:.4f} | ROC-AUC: {baseline_auc:.4f}")
print(classification_report(y_test, baseline_preds, labels=[0, 1], target_names=["Benign", "Malignant"], zero_division=0))

# Binary neural network: one sigmoid output for malignant-class probability.
keras.utils.set_random_seed(42)
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_s.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
model.fit(X_train_s, y_train.to_numpy(), epochs=60, batch_size=16, validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
          verbose=0)

nn_probs = model.predict(X_test_s, verbose=0).reshape(-1)
nn_preds = (nn_probs >= 0.5).astype(int)
nn_acc = accuracy_score(y_test, nn_preds)
nn_auc = roc_auc_score(y_test, nn_probs)
print(f"Neural Network accuracy: {nn_acc:.4f} | ROC-AUC: {nn_auc:.4f}")
print(classification_report(y_test, nn_preds, labels=[0, 1], target_names=["Benign", "Malignant"], zero_division=0))

# Choose by ROC-AUC, which accounts for ranking performance across thresholds.
results = [
    {"model": "LogisticRegression", "accuracy": round(float(baseline_acc), 4), "roc_auc": round(float(baseline_auc), 4)},
    {"model": "NeuralNetwork", "accuracy": round(float(nn_acc), 4), "roc_auc": round(float(nn_auc), 4)},
]
best_model = max(results, key=lambda item: item["roc_auc"])["model"]

model.save(BASE_DIR / "model.keras")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")
options = {
    "features": FEATURES,
    "classes": ["Benign", "Malignant"],
    "positive_class": "Malignant",
    "ranges": {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES},
    "means": {f: float(X[f].mean()) for f in FEATURES},
}
with open(BASE_DIR / "options.json", "w", encoding="utf-8") as f:
    json.dump(options, f, indent=2)
with open(BASE_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"best_model": best_model, "results": results,
               "note": "Educational only; not for clinical diagnosis."}, f, indent=2)
print("Saved: model.keras, scaler.pkl, options.json, metrics.json")
print("Note: app.py uses the trained neural network; best_model is reported for comparison only.")
