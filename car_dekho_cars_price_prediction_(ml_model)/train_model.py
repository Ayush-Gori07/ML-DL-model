"""CarDekho used car price models train karta hai, sabse kam MAE wala model save karta hai.

Run: python train_model.py
Dataset: data/car_details.csv  ("CAR DETAILS FROM CAR DEKHO" ka rename kiya hua naam)
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
# File ka extension .xls ho sakta hai, lekin andar CSV hai
df = pd.read_csv("data/car_details.csv")

df["brand"] = df["name"].str.split().str[0]
df["model"] = df["name"].str.split().str[1]

# Notebook mein drop_duplicates() assign nahi hua tha, yahan sahi se apply kiya
df = df.drop_duplicates()

# km_driven ke outliers (notebook wala 35%-65% quantile filter)
q1 = df["km_driven"].quantile(0.35)
q2 = df["km_driven"].quantile(0.65)
iqr = q2 - q1
low, high = q1 - 1.5 * iqr, q2 + 1.5 * iqr
df = df[(df["km_driven"] >= max(0, low)) & (df["km_driven"] <= high)]
print(f"Rows after cleaning: {len(df)}")

# ---------- Train / test ----------
CAT_COLS = ["fuel", "seller_type", "transmission", "owner", "brand", "model"]
NUM_COLS = ["year", "km_driven"]

X = df[CAT_COLS + NUM_COLS]
y = np.log1p(df["selling_price"])  # skew kam karne ke liye log

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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
print("(MAE / MedAE rupees mein)")

best_name = results_df.iloc[0]["model"]
print(f"\nBest model: {best_name}")

joblib.dump(fitted[best_name], "model.pkl", compress=3)
size_mb = os.path.getsize("model.pkl") / 1e6
print(f"model.pkl size: {size_mb:.1f} MB")
if size_mb > 90:
    print("WARNING: GitHub limit 100 MB hai. Chhota model (GradientBoosting/Ridge) use karo.")

# ---------- App ke dropdown options ----------
options = {
    "brand_models": {b: sorted(g["model"].dropna().unique().tolist()) for b, g in df.groupby("brand")},
    "fuel": sorted(df["fuel"].unique().tolist()),
    "seller_type": sorted(df["seller_type"].unique().tolist()),
    "transmission": sorted(df["transmission"].unique().tolist()),
    "owner": sorted(df["owner"].unique().tolist()),
    "min_year": int(df["year"].min()),
    "max_year": int(df["year"].max()),
    "max_km": int(df["km_driven"].max()),
}
with open("options.json", "w") as f:
    json.dump(options, f, indent=2)

with open("metrics.json", "w") as f:
    json.dump({"best_model": best_name, "results": results_df.to_dict("records")}, f, indent=2)

print("Saved: model.pkl, options.json, metrics.json")
