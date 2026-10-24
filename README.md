# ML / DL Projects — Streamlit Apps

6 machine-learning and deep-learning projects, each with a **Streamlit prediction app** and a training script. All apps are educational demos, not production systems.

## Projects

| # | Folder | Type | Task | Model |
|---|--------|------|------|-------|
| 1 | `Breast_cancer_prediction_(dl_model)` | Deep Learning (Keras) | Binary classification — benign vs malignant | Neural network (sigmoid) |
| 2 | `car_dekho_cars_price_prediction_(ml_model)` | ML (scikit-learn) | Regression — used car selling price (₹) | Best of Linear / Ridge / Lasso / RF / GB |
| 3 | `house_rent_price_prediction_(ml_model)` | ML (scikit-learn) | Regression — monthly house rent (₹) | Best of Linear / Ridge / Lasso / RF / GB / SVR |
| 4 | `Iris_classification_(dl_model)` | Deep Learning (Keras) | Multi-class classification — Iris species | Neural network (softmax) |
| 5 | `used_car_price_predection_(ml_model)` | ML (scikit-learn) | Regression — used car asking price (₹) | Best of Linear / Ridge / Lasso / RF / GB |
| 6 | `wine_quality_predection_(ml_model)` | ML (scikit-learn) | Binary classification — good wine (quality ≥ 7) | Random Forest (balanced) |

## Quick start (all projects)

```bash
pip install -r requirements.txt
```

Then pick a project and train + run it:

```bash
cd Breast_cancer_prediction_(dl_model)   # or any other project folder
python train_model.py
streamlit run app.py
```

If you only want one project and want to skip tensorflow, install that project's own `requirements.txt` instead of the root one.

## Project structure (each project)

```
<project_folder>/
├── app.py            # Streamlit app
├── train_model.py    # Trains model and saves artifacts
├── requirements.txt  # Per-project deps (leaner than root file)
├── README.md         # Project-specific notes
├── .gitignore        # Ignores venv, pycache, model artifacts
├── data/             # Dataset (CSV) — add your own if missing
└── (generated)       # model.pkl / model.keras, scaler.pkl,
                      # options.json, metrics.json — gitignored
```

## Generated artifacts

Running `train_model.py` creates model files inside the project folder. These are **gitignored** and should be regenerated on each machine with `python train_model.py` — do not commit them.

## Datasets

Each project's README lists its dataset source and expected CSV filename inside `data/`. Most projects already include a CSV; the Iris project and any project without a CSV need you to add the dataset manually.

## Deploying on Streamlit Cloud

For a clean deploy, use **one project per repo** so each app gets its own slim `requirements.txt` (without tensorflow for the 4 scikit-learn-only apps). If you deploy this whole folder as a single repo, Streamlit Cloud uses the root `requirements.txt` and installs tensorflow for every app.

## Important

All models are educational demos. They are not validated for real-world decision-making — do not use them for medical diagnosis, financial decisions, or anything where an incorrect prediction has real consequences.

## 🌐 Live Deployed Apps (Streamlit Cloud)

| Project | Live App |
|---------|----------|
| Breast Cancer Classification (DL) | [Open App](https://ml-dl-model-wa4dcdaffwwdkxyshxlxnh.streamlit.app/) |
| CarDekho Car Price Prediction | [Open App](https://ml-dl-model-gxtt5kjj4rts8hyjsy4vas.streamlit.app/) |
| House Rent Prediction | [Open App](https://ml-dl-model-hux5wjjkurmkmld5wq4m5t.streamlit.app/) |
| Iris Flower Classification (DL) | [Open App](https://ml-dl-model-nbpcgz2we24gmkkpqpmh6k.streamlit.app/) |
| Used Car Price Prediction | [Open App](https://ml-dl-model-vtb72i7sah2kbhk2wbuzfd.streamlit.app/) |
| Wine Quality Prediction | [Open App](https://ml-dl-model-hscmrzrc2raplwca2s83sv.streamlit.app/) |

> **Note:** Yeh URLs Streamlit Cloud deploy ke baad actual hongi. Abhi yeh placeholder hain — har app deploy karne ke baad Streamlit Cloud se milne wale actual URL se replace kar dena.
