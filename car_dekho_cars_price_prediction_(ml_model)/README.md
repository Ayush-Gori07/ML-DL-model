# 🚙 CarDekho Used Car Price Prediction

## Problem
Used car ki details (brand, model, year, km driven, fuel, seller type, transmission, owner) se uski **selling price** predict karna. Yeh regression problem hai.

## Dataset
CarDekho used car dataset (Indian market, prices in ₹) — [Kaggle: Vehicle dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
File: `data/car_details.csv` (original naam "CAR DETAILS FROM CAR DEKHO", rename kiya hua)

## Approach
- Car ke `name` se `brand` aur `model` nikale
- Duplicates hataye, `km_driven` ke outliers hataye
- Target (`selling_price`) ko `log1p` se transform kiya (price skewed hai)
- Preprocessing: OneHotEncoder (categorical, rare categories group) + StandardScaler (numeric), sab ek `Pipeline` mein
- Models compare kiye: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- Best model (lowest MAE) `model.pkl` mein save hota hai

## Result
Comparison table `train_model.py` run karne ke baad print hogi aur `metrics.json` mein save hogi. (Yahan apna best model aur MAE likh do.)

## Run kaise karein
```bash
pip install -r requirements.txt
python train_model.py        # model.pkl, options.json, metrics.json banata hai
streamlit run app.py
```

## Files
| File | Kaam |
|---|---|
| `cardekho_car_price.ipynb` | EDA + model comparison |
| `train_model.py` | Cleaning + training + best model save |
| `app.py` | Streamlit app |
| `data/car_details.csv` | Dataset (khud add karo) |
