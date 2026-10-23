"""Iris dataset pe Perceptron aur Keras neural network train karta hai, Keras model save karta hai.

Run: python train_model.py
Dataset: data/Iris.csv
Output: model.keras, scaler.pkl, options.json, metrics.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# ---------- Data ----------
df = pd.read_csv("Iris.csv")
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

X = df[FEATURES]
encoder = LabelEncoder()
y = encoder.fit_transform(df["Species"])
print(f"Rows: {len(df)} | Classes: {list(encoder.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scaler sirf train pe fit hota hai, test pe sirf transform (data leakage se bachne ke liye)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- Model 1: Perceptron (baseline) ----------
per = Perceptron(max_iter=1000, random_state=42)
per.fit(X_train_s, y_train)
per_acc = accuracy_score(y_test, per.predict(X_test_s))
print(f"\nPerceptron accuracy: {per_acc:.4f}")
print(classification_report(y_test, per.predict(X_test_s), target_names=encoder.classes_))

# ---------- Model 2: Neural network (Keras) ----------
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

model = Sequential(
    [
        Dense(16, input_shape=(4,), activation="relu"),
        Dense(8, activation="relu"),
        Dense(3, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train_s, y_train_cat, epochs=100, batch_size=8, validation_split=0.2, verbose=0)

loss, nn_acc = model.evaluate(X_test_s, y_test_cat, verbose=0)
print(f"Neural network accuracy: {nn_acc:.4f} | loss: {loss:.4f}")

# ---------- Save ----------
model.save("model.keras")
joblib.dump(scaler, "scaler.pkl")

options = {
    "features": FEATURES,
    "classes": [c.replace("Iris-", "") for c in encoder.classes_],
    "ranges": {f: [float(df[f].min()), float(df[f].max())] for f in FEATURES},
    "means": {f: round(float(df[f].mean()), 2) for f in FEATURES},
}
with open("options.json", "w") as f:
    json.dump(options, f, indent=2)

with open("metrics.json", "w") as f:
    json.dump(
        {
            "best_model": "NeuralNetwork",
            "results": [
                {"model": "Perceptron", "accuracy": round(per_acc, 4)},
                {"model": "NeuralNetwork", "accuracy": round(float(nn_acc), 4), "loss": round(float(loss), 4)},
            ],
        },
        f,
        indent=2,
    )

print("Saved: model.keras, scaler.pkl, options.json, metrics.json")
