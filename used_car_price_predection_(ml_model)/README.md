# 🚗 Used Car Price Prediction

## Problem
Used car ki details (brand, model, fuel, transmission, owner, age, km driven) se uski **asking price** predict karna. Yeh regression problem hai.

## Dataset
Used car dataset (Indian market, prices in ₹) — source link yahan daalo (Kaggle).
File: `data/used_car_dataset.csv`

## Approach
- Cleaning: `AskPrice` aur `kmDriven` se ₹ / km / commas hataye, hazaar mein convert kiya
- Missing values aur duplicates hataye
- IQR method se price aur km ke outliers hataye
- Target ko `log1p` se transform kiya (price skewed hai)
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
| `used_car_price.ipynb` | EDA + model comparison |
| `train_model.py` | Cleaning + training + best model save |
| `app.py` | Streamlit app |
| `data/used_car_dataset.csv` | Dataset (khud add karo) |
