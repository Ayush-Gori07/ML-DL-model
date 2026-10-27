# 🏠 House Rent Prediction

## Problem
Predict the **monthly rent** of a house from its details: city, locality, BHK, size, floor, furnishing, tenant preference, and other attributes. This is a regression problem.

## Dataset
House Rent Prediction Dataset (Indian cities, 4700+ listings, rent in ₹) — [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)
File: `data/House_Rent_Dataset.csv`

## Approach
- Extract `my_floor` and `total_floor` from the `Floor` column (e.g., "3 out of 5"); derive `for_family` / `for_bachelor` from `Tenant Preferred`
- Remove outliers (Rent, Size, Bathroom), clip Bathroom to 4
- Transform the target (`Rent`) using `log1p` (rent is skewed)
- Preprocessing: OneHotEncoder (categorical) + TargetEncoder (`Area Locality`, since there are many localities) + median imputer + StandardScaler (numeric), all in a single `Pipeline`
- Models compared: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, SVR
- Best model (lowest MAE) saved in `model.pkl`

## Result
The comparison table is printed after running `train_model.py` and saved in `metrics.json`. (Add your best model and MAE here.)

## Limitations
Extreme rents were removed during training, so the model is not accurate for very expensive properties.

## How to run
```bash
pip install -r requirements.txt
python train_model.py        # creates model.pkl, options.json, metrics.json
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `house_rent.ipynb` | EDA + model comparison |
| `train_model.py` | Cleaning + training + best model save |
| `app.py` | Streamlit app |
| `data/House_Rent_Dataset.csv` | Dataset (add it yourself) |
