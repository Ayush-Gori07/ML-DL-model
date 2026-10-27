# 🍷 Red Wine Quality Prediction

## Problem
Predict whether a wine is **good quality (score ≥ 7)** from its chemical properties: acidity, alcohol, sulphates, and others. This is a binary classification problem.

## Dataset
Red Wine Quality dataset (1599 samples, 11 features) — [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) / Kaggle.
File: `data/winequality-red.csv`

## Approach
- EDA: quality distribution, boxplot, correlation heatmap
- Binary labels: `≥ 7 → good (1)`, rest → `0`
- Drop low-correlation features; final 6 features: volatile acidity, citric acid, total sulfur dioxide, density, sulphates, alcohol
- Models compared: Logistic Regression, Decision Tree, SVM, KNN, Random Forest
- Final model: **Random Forest** (`class_weight="balanced"`, since good wines are only ~14%)

## Result
Accuracy and recall are available in `metrics.json` after running `train_model.py`. (Add your numbers here.)

> Note: the data is imbalanced, so do not rely on accuracy alone — check good-wine precision and recall too.

## How to run
```bash
pip install -r requirements.txt
python train_model.py        # creates model.pkl
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `wine_quality.ipynb` | EDA + model comparison |
| `train_model.py` | Final model train + save |
| `app.py` | Streamlit app |
| `data/winequality-red.csv` | Dataset (add it yourself) |
