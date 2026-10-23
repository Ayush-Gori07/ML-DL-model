# 🌸 Iris Flower Classification (Deep Learning)

## Problem
Flower ki 4 measurements (sepal/petal length aur width) se uski **species** predict karna: Iris-setosa, Iris-versicolor ya Iris-virginica. Yeh multi-class classification problem hai.

## Dataset
Iris Dataset (150 rows, 3 classes, har class ki 50 rows) — [Kaggle](https://www.kaggle.com/datasets/uciml/iris)
File: `data/Iris.csv`

## Approach
- `Id` column hataya, `Species` ko `LabelEncoder` se number banaya
- 80/20 stratified train-test split
- `StandardScaler` sirf train pe fit kiya, test pe sirf transform (data leakage se bachne ke liye)
- Baseline: sklearn `Perceptron`
- Deep learning model: Keras `Sequential` — Dense(16, relu) → Dense(8, relu) → Dense(3, softmax), `adam` optimizer, `categorical_crossentropy` loss, 100 epochs
- Final model `model.keras` mein save hota hai, scaler `scaler.pkl` mein

## Result
Comparison `train_model.py` run karne ke baad print hogi aur `metrics.json` mein save hogi. (Yahan apna Perceptron vs Neural Network accuracy likh do.)

## Limitations
Dataset bahut chhota hai (150 rows), isliye test set sirf 30 rows ka hai — accuracy thodi upar-neeche ho sakti hai random split pe.

## Run kaise karein
```bash
pip install -r requirements.txt
python train_model.py        # model.keras, scaler.pkl, options.json, metrics.json banata hai
streamlit run app.py
```

## Files
| File | Kaam |
|---|---|
| `deep_learning_model_1.ipynb` | EDA + Perceptron vs Neural Network |
| `train_model.py` | Preprocessing + training + model save |
| `app.py` | Streamlit app |
| `data/Iris.csv` | Dataset (khud add karo) |
