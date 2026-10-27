# 🚙 CarDekho Used Car Price Prediction

## Problem
Predict the **selling price** of a used car from its details: brand, model, year, kilometers driven, fuel type, seller type, transmission, and owner. This is a regression problem.

## Dataset
CarDekho used car dataset (Indian market, prices in ₹) — [Kaggle: Vehicle dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
File: `data/car_details.csv` (renamed from the original "CAR DETAILS FROM CAR DEKHO")

## Approach
- Extract `brand` and `model` from the car `name`
- Remove duplicates and `km_driven` outliers
- Transform the target (`selling_price`) using `log1p` (price is skewed)
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
| `cardekho_car_price.ipynb` | EDA + model comparison |
| `train_model.py` | Cleaning + training + best model save |
| `app.py` | Streamlit app |
| `data/car_details.csv` | Dataset (add it yourself) |
