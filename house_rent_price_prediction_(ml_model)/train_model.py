"""House rent models train karta hai, sabse kam MAE wala model save karta hai.

Run: python train_model.py
Dataset: data/House_Rent_Dataset.csv
Output: model.pkl, options.json, metrics.json
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.svm import SVR

# ---------- Feature engineering (notebook jaisa) ----------
df = pd.read_csv("data/House_Rent_Dataset.csv")

df["for_family"] = df["Tenant Preferred"].str.contains("Family", case=False).astype(int)
df["for_bachelor"] = df["Tenant Preferred"].str.contains("Bachelor", case=False).astype(int)

# Floor jaise "Ground out of 2" / "3 out of 5" -> my_floor aur total_floor
df["my_floor"] = df["Floor"].str.split().str[0]
df["total_floor"] = df["Floor"].str.split().str[-1]

# ---------- Cleaning ----------
df = df[df["Bathroom"] < 7]

# total_floor jahan number nahi hai (text), woh rows hatao
num = pd.to_numeric(df["total_floor"], errors="coerce")
bad = num.isna() & df["total_floor"].notna()
df = df[~bad].copy()
df["total_floor"] = pd.to_numeric(df["total_floor"])

df["my_floor"] = df["my_floor"].replace({"Ground": 0, "Upper": -1, "Lower": -2})
df["my_floor"] = pd.to_numeric(df["my_floor"])


def remove_outliers(data, col, lo_q, hi_q):
    q1, q2 = data[col].quantile(lo_q), data[col].quantile(hi_q)
    iqr = q2 - q1
    low, high = max(0, q1 - 1.5 * iqr), q2 + 1.5 * iqr
    return data[(data[col] >= low) & (data[col] <= high)]


df = remove_outliers(df, "Rent", 0.30, 0.70)   # notebook wala 30%-70%
df = remove_outliers(df, "Size", 0.25, 0.75)
df = df[np.log1p(df["Rent"]) >= 8]
df["Bathroom"] = df["Bathroom"].clip(upper=4)
df = df[df["Size"] >= 100]
print(f"Rows after cleaning: {len(df)}")

# ---------- Train / test ----------
CAT_COLS = ["Area Type", "City", "Furnishing Status", "Point of Contact"]
LOC_COL = ["Area Locality"]
NUM_COLS = ["BHK", "Size", "Bathroom", "for_family", "for_bachelor", "my_floor", "total_floor"]

X = df[CAT_COLS + LOC_COL + NUM_COLS]
y = np.log1p(df["Rent"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", drop="first", min_frequency=10), CAT_COLS),
        ("loc", TargetEncoder(smooth="auto"), LOC_COL),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_COLS),
    ]
)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(kernel="rbf", C=10, epsilon=0.1),
}

results, fitted = [], {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    actual, predicted = np.expm1(y_test), np.expm1(pred)
    results.append(
        {
            "model": name,
            "R2_log": round(r2_score(y_test, pred), 4),
            "MAE": round(mean_absolute_error(actual, predicted), 2),
            "MedAE": round(median_absolute_error(actual, predicted), 2),
        }
    )
    fitted[name] = pipe

results_df = pd.DataFrame(results).sort_values("MAE")
print(results_df.to_string(index=False))
print("(MAE / MedAE rupees per month mein)")

best_name = results_df.iloc[0]["model"]
print(f"\nBest model: {best_name}")

joblib.dump(fitted[best_name], "model.pkl", compress=3)
size_mb = os.path.getsize("model.pkl") / 1e6
print(f"model.pkl size: {size_mb:.1f} MB")
if size_mb > 90:
    print("WARNING: GitHub limit 100 MB hai. Chhota model (GradientBoosting/Ridge) use karo.")

# ---------- App ke dropdown options ----------
options = {
    "city_localities": {c: sorted(g["Area Locality"].dropna().unique().tolist()) for c, g in df.groupby("City")},
    "area_type": sorted(df["Area Type"].unique().tolist()),
    "furnishing": sorted(df["Furnishing Status"].unique().tolist()),
    "contact": sorted(df["Point of Contact"].unique().tolist()),
    "tenant": sorted(df["Tenant Preferred"].unique().tolist()),
    "max_bhk": int(df["BHK"].max()),
    "min_size": int(df["Size"].min()),
    "max_size": int(df["Size"].max()),
    "min_rent": int(df["Rent"].min()),
    "max_rent": int(df["Rent"].max()),
}
with open("options.json", "w") as f:
    json.dump(options, f, indent=2)

with open("metrics.json", "w") as f:
    json.dump({"best_model": best_name, "results": results_df.to_dict("records")}, f, indent=2)

print("Saved: model.pkl, options.json, metrics.json")
