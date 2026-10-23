"""Wine quality model train karke model.pkl aur metrics.json save karta hai.

Run: python train_model.py
Dataset: data/winequality-red.csv (UCI / Kaggle "Red Wine Quality")
"""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Notebook mein jo features bache the (baaki drop kiye the)
FEATURES = [
    "volatile acidity",
    "citric acid",
    "total sulfur dioxide",
    "density",
    "sulphates",
    "alcohol",
]

df = pd.read_csv("data/winequality-red.csv")

# Notebook jaisa: quality >= 7 -> good (1), baaki -> not good (0)
df["quality"] = df["quality"].map(lambda x: 0 if x <= 6 else 1)

X = df[FEATURES]
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Good wine kam hain (~14%), isliye class_weight="balanced"
model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")

# Generate options.json for app consistency with other projects.
options = {
    "features": FEATURES,
    "classes": ["Not good", "Good"],
    "ranges": {f: [float(df[f].min()), float(df[f].max())] for f in FEATURES},
    "means": {f: round(float(df[f].mean()), 4) for f in FEATURES},
}
with open("options.json", "w", encoding="utf-8") as f:
    json.dump(options, f, indent=2)

with open("metrics.json", "w") as f:
    json.dump(
        {
            "accuracy": round(acc, 4),
            "good_precision": round(report["1"]["precision"], 4),
            "good_recall": round(report["1"]["recall"], 4),
            "features": FEATURES,
        },
        f,
        indent=2,
    )
print("Saved: model.pkl, options.json, metrics.json")
