# 🌸 Iris Flower Classification (Deep Learning)

## Problem
Predict the **species** of an iris flower from 4 measurements (sepal and petal length and width): Iris-setosa, Iris-versicolor, or Iris-virginica. This is a multi-class classification problem.

## Dataset
Iris Dataset (150 rows, 3 classes, 50 rows per class) — [Kaggle](https://www.kaggle.com/datasets/uciml/iris)
File: `data/Iris.csv`

## Approach
- Remove the `Id` column, encode `Species` to integers using `LabelEncoder`
- 80/20 stratified train-test split
- Fit `StandardScaler` on training data only, transform test data only (to avoid data leakage)
- Baseline: sklearn `Perceptron`
- Deep learning model: Keras `Sequential` — Dense(16, relu) → Dense(8, relu) → Dense(3, softmax), `adam` optimizer, `categorical_crossentropy` loss, 100 epochs
- Final model saved in `model.keras`, scaler in `scaler.pkl`

## Result
The comparison is printed after running `train_model.py` and saved in `metrics.json`. (Add your Perceptron vs Neural Network accuracy here.)

## Limitations
The dataset is small (150 rows), so the test set is only 30 rows — accuracy may vary somewhat depending on the random split.

## How to run
```bash
pip install -r requirements.txt
python train_model.py        # creates model.keras, scaler.pkl, options.json, metrics.json
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `deep_learning_model_1.ipynb` | EDA + Perceptron vs Neural Network |
| `train_model.py` | Preprocessing + training + model save |
| `app.py` | Streamlit app |
| `data/Iris.csv` | Dataset (add it yourself) |
