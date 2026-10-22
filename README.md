# AI Disease Risk Prediction Dashboard

A modern clinical decision-support web application for multi-disease risk assessment. The dashboard allows clinicians and users to select a condition (Type 2 Diabetes or Breast Cancer), input physiological measurements, and receive machine learning-based risk probabilities alongside transparent, explainable feature attribution charts.

> **Disclaimer:** Educational demo only. Not a medical diagnosis.

---

## Features

- **Multi-Disease Screening:** Supports Type 2 Diabetes and Wisconsin Diagnostic Breast Cancer risk prediction.
- **Dynamic Physiological Form:** Client-side range validation, biologically sane limits, and preset clinical patient profiles (high-risk and low-risk).
- **Explainable AI (XAI):** Visual horizontal bar chart illustrating the top 5 factors driving or lowering risk for each prediction based on standardized regression coefficients.
- **Probability Calibration Display:** Unambiguous probability presentation clamped to `>99%` or `<1%` to prevent unrealistic 100% or 0% certainty indications.
- **Empirical Evaluation Tab:** Live model evaluation displays holdout test metrics (Accuracy, Precision, Recall, F1, ROC-AUC), 2×2 diagnostic confusion matrices, and 50-run repeated stratified cross-validation benchmarks.
- **Interactive Dataset Explorer:** In-app inspection of dataset provenance, record counts, feature definitions, and class balance.

---

## Tech Stack

- **Backend:** Python managed with `uv`, FastAPI, Uvicorn, Scikit-learn, Pandas, NumPy, Joblib, Pydantic, Pytest.
- **Frontend:** React, Vite, Recharts, Lucide-React, Vanilla CSS design system with Inter typography.
- **Machine Learning:** Scikit-learn Pipelines (`SimpleImputer(median)` ➔ `StandardScaler()` ➔ `LogisticRegression(max_iter=1000)`).

---

## Project Structure

```
healthrisk-dashboard/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── train.py
│   ├── models/
│   │   ├── diabetes.joblib
│   │   ├── diabetes_metrics.json
│   │   ├── breast_cancer.joblib
│   │   └── breast_cancer_metrics.json
│   ├── tests/
│   │   ├── conftest.py
│   │   └── test_api.py
│   ├── pyproject.toml
│   └── uv.lock
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
├── docs/
│   └── screenshots/
│       └── .gitkeep
├── .gitignore
└── README.md
```

---

## Datasets

| Disease | Dataset | Source | Records | Features | Target |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Diabetes** | [Pima Indians Diabetes](https://www.openml.org/search?type=data&id=37) | National Institute of Diabetes and Digestive and Kidney Diseases via OpenML | 768 | 8 | Diabetes diagnosis (1 = diabetic, 0 = non-diabetic) |
| **Breast Cancer** | [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | UCI Machine Learning Repository via Scikit-learn | 569 | 10 | Tumor malignancy status (1 = malignant, 0 = benign) |

*Preprocessing note:* In the Diabetes dataset, biologically impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI were treated as missing values and imputed with the training median; in Breast Cancer, only the 10 core mean nuclear morphology attributes were extracted and the target was calibrated so 1 represents malignant.

---

## Model Evaluation Results

Empirical results computed from direct training pipeline runs:

### Holdout Test Performance (80/20 Stratified Split)

| Disease | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Confusion Matrix `[TN, FP / FN, TP]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diabetes** | 70.78% | 60.00% | 50.00% | 0.5455 | 0.8130 | `[82, 18 / 27, 27]` |
| **Breast Cancer** | 92.98% | 88.64% | 92.86% | 0.9070 | 0.9841 | `[67, 5 / 3, 39]` |

### Repeated Stratified Cross-Validation (5 Folds × 10 Repeats = 50 Runs)

| Disease | Model Candidate | Mean Accuracy | Mean F1-Score | Mean ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Diabetes** | Baseline (Majority Class) | 65.10% ± 0.20% | 0.000 ± 0.000 | 0.500 ± 0.000 |
| | **Logistic Regression (Deployed)** | **76.70% ± 2.90%** | **0.624 ± 0.054** | **0.835 ± 0.028** |
| | Random Forest (200 trees) | 76.50% ± 2.50% | 0.641 ± 0.042 | 0.829 ± 0.027 |
| **Breast Cancer** | Baseline (Majority Class) | 62.70% ± 0.40% | 0.000 ± 0.000 | 0.500 ± 0.000 |
| | **Logistic Regression (Deployed)** | **93.70% ± 2.00%** | **0.912 ± 0.029** | **0.985 ± 0.006** |
| | Random Forest (200 trees) | 93.70% ± 2.40% | 0.915 ± 0.033 | 0.983 ± 0.009 |

---

## How to Run

### Backend Setup
```bash
cd backend
uv sync
uv run python -m app.train
uv run pytest -v
uv run uvicorn app.main:app --port 8000
```
Backend API will be accessible at `http://localhost:8000` (documentation at `http://localhost:8000/docs`).

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend interface will be live at `http://localhost:5173`.

---

## Screenshots

- `docs/screenshots/form.png` — Dynamic clinical parameters input form and preset selectors.
- `docs/screenshots/result.png` — Risk percentage badge, gauge, and Recharts top contributing factors chart.
- `docs/screenshots/performance.png` — Holdout evaluation metrics, cross-validation benchmark table, and 2×2 confusion matrix.

---

## Limitations

1. **Small Datasets:** The Pima Diabetes ($N=768$) and Wisconsin Breast Cancer ($N=569$) cohorts are small benchmark datasets and may not reflect broader global patient demographics.
2. **Not Clinically Validated:** The application is built for laboratory and academic exploration. It has not undergone formal clinical trial verification and must not supersede qualified medical practitioners.
3. **Diabetes Recall is Modest:** Diabetes holdout recall is 50.00%, meaning subtle borderline diabetic cases may be missed without comprehensive longitudinal biomarker panels (such as HbA1c).

---

## Team

| Name | Roll No | Role |
| :--- | :--- | :--- |
| Student 1 | [Roll No 1] | Healthcare Data & Preprocessing Lead |
| Student 2 | [Roll No 2] | Machine Learning & Model Evaluation Lead |
| Student 3 | [Roll No 3] | Backend API & Validation Engineer |
| Student 4 | [Roll No 4] | Frontend UI/UX Developer |
