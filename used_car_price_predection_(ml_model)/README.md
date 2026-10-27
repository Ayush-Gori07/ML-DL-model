# 🚗 Used Car Price Prediction

## Problem
Predict the **asking price** of a used car from its details: brand, model, fuel type, transmission, owner, age, and kilometers driven. This is a regression problem.

## Dataset
Used car dataset (Indian market, prices in ₹) — add the source link here (Kaggle).
File: `data/used_car_dataset.csv`

## Approach
- Cleaning: remove ₹, "km", and commas from `AskPrice` and `kmDriven`, convert to thousands
- Remove missing values and duplicates
- Remove price and km outliers using the IQR method
- Transform the target using `log1p` (price is skewed)
- Preprocessing: OneHotEncoder (categorical, rare categories grouped) + StandardScaler (numeric), all in a single `Pipeline`
- Models compared: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
- Best model (lowest MAE) saved in `model.pkl`

## Result
The comparison table is printed after running `train_model.py` and saved in `metrics.json`. (Add your best model and MAE here.)

## How to run
```bash
pip install -r requirements.txt
python train_model.py        # creates model.pkl, options.json, metrics.json
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `used_car_price.ipynb` | EDA + model comparison |
| `train_model.py` | Cleaning + training + best model save |
| `app.py` | Streamlit app |
| `data/used_car_dataset.csv` | Dataset (add it yourself) |
