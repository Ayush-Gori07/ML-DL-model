# 🏠 House Rent Prediction

## Problem
Ghar ki details (city, locality, BHK, size, floor, furnishing, tenant preference, etc.) se uska **monthly rent** predict karna. Yeh regression problem hai.

## Dataset
House Rent Prediction Dataset (Indian cities, 4700+ listings, rent in ₹) — [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)
File: `data/House_Rent_Dataset.csv`

## Approach
- `Floor` column ("3 out of 5") se `my_floor` aur `total_floor` nikale; `Tenant Preferred` se `for_family` / `for_bachelor`
- Outliers hataye (Rent, Size, Bathroom), Bathroom ko 4 pe clip kiya
- Target (`Rent`) ko `log1p` se transform kiya (rent skewed hai)
- Preprocessing: OneHotEncoder (categorical) + TargetEncoder (`Area Locality`, kyunki bahut saari localities hain) + median imputer + StandardScaler (numeric), sab ek `Pipeline` mein
- Models compare kiye: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR
- Best model (lowest MAE) `model.pkl` mein save hota hai

## Result
Comparison table `train_model.py` run karne ke baad print hogi aur `metrics.json` mein save hogi. (Yahan apna best model aur MAE likh do.)

## Limitations
Training mein extreme rents hataye gaye, isliye model bahut mehngi properties ke liye accurate nahi hai.

## Run kaise karein
```bash
pip install -r requirements.txt
python train_model.py        # model.pkl, options.json, metrics.json banata hai
streamlit run app.py
```

## Files
| File | Kaam |
|---|---|
| `house_rent.ipynb` | EDA + model comparison |
| `train_model.py` | Cleaning + training + best model save |
| `app.py` | Streamlit app |
| `data/House_Rent_Dataset.csv` | Dataset (khud add karo) |
