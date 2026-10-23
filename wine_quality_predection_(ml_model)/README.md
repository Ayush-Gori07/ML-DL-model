# 🍷 Red Wine Quality Prediction

## Problem
Wine ke chemical properties (acidity, alcohol, sulphates, etc.) se predict karna ki wine **good quality (score ≥ 7)** hai ya nahi. Yeh binary classification problem hai.

## Dataset
Red Wine Quality dataset (1599 samples, 11 features) — [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) / Kaggle.
File: `data/winequality-red.csv`

## Approach
- EDA: quality distribution, boxplot, correlation heatmap
- Quality ko binary banaya: `≥ 7 → good (1)`, baaki `0`
- Weak correlation wale features drop kiye; final 6 features: volatile acidity, citric acid, total sulfur dioxide, density, sulphates, alcohol
- Models compare kiye: Logistic Regression, Decision Tree, SVM, KNN, Random Forest
- Final model: **Random Forest** (`class_weight="balanced"`, kyunki good wine sirf ~14% hain)

## Result
Accuracy aur recall `train_model.py` run karne ke baad `metrics.json` mein milenge. (Yahan apna number likh dena.)

> Note: data imbalanced hai, isliye sirf accuracy mat dekho — good-wine ka precision/recall bhi dekho.

## Run kaise karein
```bash
pip install -r requirements.txt
python train_model.py        # model.pkl banata hai
streamlit run app.py
```

## Files
| File | Kaam |
|---|---|
| `wine_quality.ipynb` | EDA + model comparison |
| `train_model.py` | Final model train + save |
| `app.py` | Streamlit app |
| `data/winequality-red.csv` | Dataset (khud add karo) |
