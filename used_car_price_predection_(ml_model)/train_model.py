"""Used car price models train karta hai, sabse kam MAE wala model save karta hai.

Run: python train_model.py
Dataset: data/used_car_dataset.csv
Output: model.pkl, options.json, metrics.json
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- Data cleaning (notebook jaisa) ----------
df = pd.read_csv("data/used_car_dataset.csv")
df = df.drop(columns=["AdditionInfo", "PostedDate", "Year"])

df["AskPrice"] = (
    df["AskPrice"].str.replace("₹ ", "", regex=False).str.replace(",", "", regex=False).astype(float)
)
df["kmDriven"] = (
    df["kmDriven"].str.replace(" km", "", regex=False).str.replace(",", "", regex=False).astype(float)
)

# Price aur km ko hazaar mein convert kiya
df["kmDriven"] = df["kmDriven"] / 1000
df["AskPrice"] = df["AskPrice"] / 1000

df = df.dropna().drop_duplicates()


def remove_outliers(data, col):
    q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    iqr = q3 - q1
    low, high = max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
    return data[(data[col] >= low) & (data[col] <= high)]


df = remove_outliers(df, "AskPrice")
df = remove_outliers(df, "kmDriven")
df = df[df["Age"] != 25]  # notebook mein yeh rows hataye the

# ---------- Train / test ----------
CAT_COLS = ["Brand", "model", "FuelType", "Transmission", "Owner"]
NUM_COLS = ["Age", "kmDriven"]

X = df[CAT_COLS + NUM_COLS]
y = np.log1p(df["AskPrice"])  # skew kam karne ke liye log

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ]
)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

results, fitted = [], {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    actual_price, pred_price = np.expm1(y_test), np.expm1(pred)
    results.append(
        {
            "model": name,
            "R2_log": round(r2_score(y_test, pred), 4),
            "MAE": round(mean_absolute_error(actual_price, pred_price), 2),
            "MedAE": round(median_absolute_error(actual_price, pred_price), 2),
        }
    )
    fitted[name] = pipe

results_df = pd.DataFrame(results).sort_values("MAE")
print(results_df.to_string(index=False))
print("(MAE / MedAE hazaar rupees mein)")

best_name = results_df.iloc[0]["model"]
print(f"\nBest model: {best_name}")

joblib.dump(fitted[best_name], "model.pkl", compress=3)
size_mb = os.path.getsize("model.pkl") / 1e6
print(f"model.pkl size: {size_mb:.1f} MB")
if size_mb > 90:
    print("WARNING: GitHub limit 100 MB hai. Chhota model (GradientBoosting/Ridge) use karo.")

# ---------- App ke dropdown options ----------
options = {
    "brand_models": {b: sorted(g["model"].unique().tolist()) for b, g in df.groupby("Brand")},
    "fuel": sorted(df["FuelType"].unique().tolist()),
    "transmission": sorted(df["Transmission"].unique().tolist()),
    "owner": sorted(df["Owner"].unique().tolist()),
    "max_age": int(df["Age"].max()),
    "max_km": round(float(df["kmDriven"].max()), 1),
}
with open("options.json", "w") as f:
    json.dump(options, f, indent=2)

with open("metrics.json", "w") as f:
    json.dump({"best_model": best_name, "results": results_df.to_dict("records")}, f, indent=2)

print("Saved: model.pkl, options.json, metrics.json")
