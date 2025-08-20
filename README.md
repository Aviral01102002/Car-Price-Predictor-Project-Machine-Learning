# Car Price Predictor — Machine Learning Project

Predict used car prices with a clean, reproducible ML pipeline. This repo covers data prep, EDA, feature engineering, model training (Linear/Regularized, Tree‑based, and Ensemble models), evaluation, and optional API/UI deployment.

---

## ✨ Highlights

* End‑to‑end pipeline using **scikit‑learn** & **pandas**
* Handles missing values, outliers, categorical encoding, scaling
* Trains multiple models: **Linear Regression**, **Ridge/Lasso**, **Random Forest**, **XGBoost** (optional)
* Proper evaluation with **K‑Fold CV**, **MAE/MSE/RMSE/R²**
* Feature importance + **SHAP** (optional) for explainability
* CLI scripts for training/inference; **FastAPI** endpoint; **Streamlit** demo app
* Reproducible results with a fixed **random\_state** and **requirements.txt**

---

## 🗂️ Repository Structure

```
Car-Price-Predictor-Project-Machine-Learning/
├─ data/
│  ├─ raw/               # put original dataset(s) here (not tracked)
│  ├─ interim/           # intermediate files after cleaning/featurization
│  └─ processed/         # final train/test sets
├─ notebooks/
│  ├─ 01_eda.ipynb
│  └─ 02_modeling.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ config.py          # paths, constants, params
│  ├─ data_prep.py       # load/clean/feature-engineer
│  ├─ train.py           # training pipeline
│  ├─ evaluate.py        # metrics, plots
│  ├─ predict.py         # batch/single prediction helpers
│  ├─ api.py             # FastAPI app
│  └─ app_streamlit.py   # Streamlit demo
├─ models/
│  ├─ model.pkl          # trained model
│  └─ encoder.pkl        # fitted encoders/scalers
├─ reports/
│  ├─ figures/
│  │  ├─ eda_distributions.png
│  │  ├─ feature_importance.png
│  │  └─ residuals_plot.png
│  └─ metrics.json
├─ requirements.txt
├─ .gitignore
└─ README.md
```

> **Note:** `data/raw` is git‑ignored. Place your CSV(s) there.

---

## 📦 Installation

```bash
# 1) Clone
git clone https://github.com/<your-username>/Car-Price-Predictor-Project-Machine-Learning.git
cd Car-Price-Predictor-Project-Machine-Learning

# 2) Create & activate virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# or (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt
```

**Minimal `requirements.txt`**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
joblib
xgboost           # optional
fastapi           # optional API
uvicorn[standard] # optional API server
streamlit         # optional UI
shap              # optional explainability
```

---

## 🧾 Dataset

You can use any used‑car dataset (e.g., Kaggle). Expected columns (rename as needed):

| Column         | Type     | Description                 |
| -------------- | -------- | --------------------------- |
| `price`        | numeric  | Target variable (car price) |
| `year`         | int      | Manufacturing year          |
| `brand`        | category | Brand/Make                  |
| `model`        | category | Model name                  |
| `km_driven`    | int      | Odometer reading            |
| `fuel`         | category | Petrol/Diesel/CNG/Electric  |
| `transmission` | category | Manual/Automatic            |
| `owner`        | category | First/Second/etc. owner     |
| `mileage`      | float    | kmpl (if available)         |
| `engine`       | float    | Engine CC (if available)    |
| `max_power`    | float    | BHP/PS (if available)       |
| `location`     | category | City/State                  |

Place your CSV as `data/raw/cars.csv` (or update `config.py`).

---

## 🧹 Data Cleaning & Feature Engineering

* Remove duplicates and impossible values (e.g., `price <= 0`, `year < 1990` if out‑of‑scope)
* Impute missing numeric with **median**; categorical with **most frequent**
* Create features: `car_age = current_year - year`, `km_per_year = km_driven / car_age` (safe divide)
* Encode categoricals with **One‑Hot** or **Target** encoding
* Scale numeric features with **StandardScaler** (for linear models)
* Optional: log‑transform `price` to stabilize variance

These steps are implemented in `src/data_prep.py` via a **ColumnTransformer + Pipeline**.

---

## ▶️ Quickstart (Train → Evaluate → Predict)

### 1) Prepare data

```bash
python -m src.data_prep --input data/raw/cars.csv --train data/processed/train.csv --test data/processed/test.csv --test_size 0.2 --random_state 42
```

### 2) Train models

```bash
python -m src.train --train data/processed/train.csv --model_out models/model.pkl --encoder_out models/encoder.pkl --algo rf --random_state 42
# --algo options: lr | ridge | lasso | rf | xgb
```

### 3) Evaluate

```bash
python -m src.evaluate --test data/processed/test.csv --model models/model.pkl --encoder models/encoder.pkl --out reports/metrics.json --plots_dir reports/figures
```

Metrics saved to `reports/metrics.json`. Example fields:

```json
{
  "MAE": 74512.3,
  "RMSE": 112345.6,
  "R2": 0.86
}
```

### 4) Predict (batch)

```bash
python -m src.predict --input data/processed/test.csv --model models/model.pkl --encoder models/encoder.pkl --out predictions.csv
```

### 5) Predict (single JSON)

```bash
python -m src.predict --json '{"year":2017,"brand":"Hyundai","model":"i20","km_driven":48000,"fuel":"Petrol","transmission":"Manual","owner":"First","mileage":18.6,"engine":1197,"max_power":82,"location":"Bengaluru"}' \
  --model models/model.pkl --encoder models/encoder.pkl
```

---

## 📊 Model Choices & Rationale

* **Linear / Ridge / Lasso**: strong baseline, interpretable
* **Random Forest**: robust to non‑linearities & outliers, low tuning effort
* **XGBoost** (optional): usually best accuracy with careful tuning

We use **cross‑validation** and keep a **hold‑out test set** for honest generalization estimates.

---

## 🔍 Explainability (Optional)

Enable **SHAP** to understand feature impact:

```bash
python -m src.evaluate --test data/processed/test.csv --model models/model.pkl --encoder models/encoder.pkl --shap --plots_dir reports/figures
```

This produces plots like `shap_summary.png` to see which features drive price.

---

## 🌐 REST API (FastAPI)

Run a lightweight prediction service:

```bash
uvicorn src.api:app --reload --port 8000
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2018,
    "brand": "Maruti",
    "model": "Swift",
    "km_driven": 35000,
    "fuel": "Petrol",
    "transmission": "Manual",
    "owner": "First",
    "mileage": 20.4,
    "engine": 1197,
    "max_power": 82,
    "location": "Delhi"
  }'
```

Response:

```json
{"price": 512345.67}
```

---

## 🖥️ Streamlit App (Optional)

```bash
streamlit run src/app_streamlit.py
```

Provides a simple UI to enter car details and see predicted price plus feature importance.

---

## 🧪 Testing

* Add unit tests under `tests/` for data prep and prediction functions
* Example command with `pytest`:

```bash
pytest -q
```

---

## 📈 Tips for Better Scores

* Remove rare categories or group them as `Other`
* Tune RF/XGB hyperparameters (e.g., `n_estimators`, `max_depth`, `learning_rate`)
* Try target encoding for `brand/model`
* Consider region‑wise models if data spans many cities

---

## 🔒 Reproducibility

* All scripts accept `--random_state 42`
* Pin exact package versions in `requirements.txt`
* Save final artifacts to `models/` and logs/figures to `reports/`

---

## 🧠 FAQ

**Q: My dataset columns differ.**  Update column names in `config.py` or pass mapping flags.

**Q: Predictions look too large/small.**  Check units (e.g., `km_driven` in km), handle extreme outliers, consider log‑price training.

**Q: How do I export to on‑device (mobile/web)?**  Serve with FastAPI + lightweight client, or export to ONNX.

---

## 🧩 Example `config.py`

```python
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw" / "cars.csv"
TRAIN_OUT = BASE_DIR / "data" / "processed" / "train.csv"
TEST_OUT  = BASE_DIR / "data" / "processed" / "test.csv"
MODEL_OUT = BASE_DIR / "models" / "model.pkl"
ENCODER_OUT = BASE_DIR / "models" / "encoder.pkl"
TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2
```

---

## 📜 License

Specify your license (e.g., MIT). Example:

```
MIT License — see LICENSE file for details.
```

---

## 🙌 Acknowledgements

* Thanks to open datasets (e.g., Kaggle) and the scikit‑learn community

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue to discuss major changes first.

---

## 📮 Contact

**Your Name** — Aviral Yadav.
