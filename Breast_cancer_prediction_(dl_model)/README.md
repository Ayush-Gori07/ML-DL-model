# 🩺 Breast Cancer Classification (Deep Learning)

## Project
A binary-classification learning project using numeric cell-measurement features to estimate whether a sample belongs to the **benign** or **malignant** class. This is an educational demo, not a medical device or diagnostic tool.

## Dataset
The training script accepts a Wisconsin Diagnostic Breast Cancer CSV named `data.csv` in this folder or `data/data.csv` inside the `data/` folder. The CSV should contain a `diagnosis` column with `B`/`M` or `Benign`/`Malignant`, plus numeric feature columns. It automatically drops common ID and unnamed columns.

If the CSV is absent, the script falls back to scikit-learn's bundled Wisconsin breast-cancer dataset. The fallback makes the project runnable without manually downloading a CSV, although its feature names differ from the original Kaggle CSV.

## Approach
- Binary labels: `0 = Benign`, `1 = Malignant`.
- 80/20 stratified train-test split.
- `StandardScaler` is fitted only on training data to avoid data leakage.
- Baseline: scikit-learn Logistic Regression.
- Deep learning: Keras dense network with ReLU layers, dropout, and one sigmoid output; binary cross-entropy loss and early stopping.
- Models are compared using test accuracy and ROC-AUC; metrics are saved in `metrics.json`.
- The Streamlit app collects feature values and displays the neural network's class probability.

## Setup and run
Run these commands from this project directory:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The first training run creates `model.keras`, `scaler.pkl`, `options.json`, and `metrics.json`. These generated files are intentionally ignored by Git; recreate them with `python train_model.py` on a new machine.

## Files
| File | Purpose |
|---|---|
| `breast_cancer_dl_model.ipynb` | Notebook for exploration and neural-network experiments (keep your existing notebook here) |
| `train_model.py` | Data loading, preprocessing, baseline + neural-network training, and artifact saving |
| `app.py` | Streamlit prediction demo |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes virtual environments, caches, and generated model artifacts |
| `data.csv` or `data/data.csv` | Optional local dataset |

## Important limitations
- Test accuracy and ROC-AUC are estimates from one split, not proof of clinical performance.
- Input measurements must match the feature columns used during training and must be in the expected units.
- A displayed probability is a model output, not a person's clinical cancer risk.
- Do not use this project to make screening, diagnosis, or treatment decisions. Clinical interpretation requires qualified healthcare professionals and validated testing.
