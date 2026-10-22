<div align="center">

# 🩺 AI-Based Disease Risk Prediction Dashboard

### A modern, explainable clinical decision-support web application for multi-disease risk assessment.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.159-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/Vite-8.3-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/uv-Astral-DE5FE9?style=for-the-badge" alt="uv" />
  <img src="https://img.shields.io/badge/Status-Educational%20Demo-orange?style=for-the-badge" alt="Status" />
</p>

</div>

> 🛡️ **Educational demo only. Not a medical diagnosis.**

---

<div align="center">
  <img src="docs/screenshots/02-diabetes-high-risk.png" alt="AI Disease Risk Prediction Dashboard Hero" width="100%" />
</div>

---

## 🌟 Overview

The **AI-Based Disease Risk Prediction Dashboard** provides an interactive clinical decision-support interface for assessing patient risk across two major clinical conditions: **Type 2 Diabetes** and **Breast Cancer**. The application couples Scikit-learn pipelines with a high-performance FastAPI backend and a responsive React frontend, delivering evaluated risk probabilities and transparent feature attribution breakdowns. By computing standardized regression coefficient contributions, each prediction is paired with a visual explanation illustrating which physiological markers increase or reduce disease likelihood.

---

## ✨ Key Features

- 🎯 **Multi-Disease Screening:** Instantly switch between Type 2 Diabetes (Pima Indians) and Wisconsin Diagnostic Breast Cancer risk models.
- 📋 **Dynamic Validated Form:** Dynamically rendered input fields featuring unit labels, training range hints, and real-time client-side validation against biological sane limits.
- ⚡ **Preset Patient Profiles:** Quick one-click loaders for verified test-set high-risk and low-risk patient cases.
- 📈 **Calibrated Risk Tiers:** Evaluates risk probability categorized into distinct tiers (Low &lt;33%, Moderate 33–66%, High &gt;66%) with probability clamping (`>99%` or `<1%`) to avoid unrealistic 100% or 0% certainty indications.
- 🔍 **Explainable AI (XAI) Attribution:** Recharts horizontal bar chart detailing the top 5 factors influencing the score (red for risk drivers, green for protective effects).
- ⚠️ **Out-of-Range Warnings:** Informational notices flag inputs that fall outside training distribution limits while remaining physiologically possible.
- 📊 **Model Performance & Benchmarks:** Dedicated evaluation tab presenting holdout test metrics, 2×2 diagnostic confusion matrices, and 50-run repeated stratified cross-validation comparisons (Baseline vs. Logistic Regression vs. Random Forest).
- 🗄️ **Interactive Dataset Explorer:** In-app inspection card displaying dataset provenance, record counts, feature sets, and class balance.
- 🧪 **Automated Test Suite:** Comprehensive Pytest test suite covering endpoints, schema validation, risk monotonicity, and error handling.

---

## 📸 Interface Gallery

<table>
  <tr>
    <td width="50%">
      <img src="docs/screenshots/01-dashboard-empty.png" alt="Default Dashboard State" width="100%" />
      <p align="center"><b>Default Dashboard View</b><br/>Input form with clinical defaults and initial prompt state.</p>
    </td>
    <td width="50%">
      <img src="docs/screenshots/02-diabetes-high-risk.png" alt="Diabetes High-Risk Prediction" width="100%" />
      <p align="center"><b>Diabetes High-Risk Assessment</b><br/>High-risk prediction with risk gauge, badges, and factor drivers.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="docs/screenshots/03-diabetes-low-risk.png" alt="Diabetes Low-Risk Prediction" width="100%" />
      <p align="center"><b>Diabetes Low-Risk Assessment</b><br/>Low-risk outcome displaying protective feature contributions.</p>
    </td>
    <td width="50%">
      <img src="docs/screenshots/04-breast-cancer-result.png" alt="Breast Cancer High-Risk Assessment" width="100%" />
      <p align="center"><b>Breast Cancer Assessment</b><br/>Evaluated malignancy risk with clamped probability display.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="docs/screenshots/05-model-performance.png" alt="Model Performance & Evaluation" width="100%" />
      <p align="center"><b>Model Performance Tab</b><br/>Holdout metrics cards, cross-validation table, and 2×2 confusion matrix.</p>
    </td>
    <td width="50%">
      <img src="docs/screenshots/06-about-the-data.png" alt="About the Dataset Explorer" width="100%" />
      <p align="center"><b>Dataset Information Panel</b><br/>Provenance, record counts, and clinical class distribution.</p>
    </td>
  </tr>
</table>

---

## 🏗️ System Architecture & How It Works

```mermaid
flowchart TD
    subgraph Client["💻 Client Layer (React 19 + Vite + Recharts)"]
        User(["👤 Clinician / User"])
        InputForm["📋 Dynamic Input Form<br/>• Real-time range validation<br/>• High/Low-risk preset loader"]
        VizEngine["📊 Visualization Engine<br/>• Calibrated Risk Gauge (&gt;99% / &lt;1%)<br/>• Recharts Top-5 Feature Attribution Bar Chart"]
        User -->|1. Enters health metrics| InputForm
        InputForm -->|2. Submits payload| RequestBus["⚡ Axios/Fetch HTTP Client"]
    end

    subgraph Gateway["🚀 API Gateway (FastAPI + Pydantic)"]
        Router["🛣️ REST Router<br/>POST /api/predict/{disease}"]
        Validator["🛡️ Physiological Validator<br/>• Sane bounds check (422 rejection)<br/>• Out-of-distribution warning tagger"]
        RequestBus -->|3. JSON Payload| Router
        Router --> Validator
    end

    subgraph MLEngine["🧠 Scikit-Learn Inference Pipeline"]
        Imputer["🔧 SimpleImputer<br/>(Median replacement for missing 0s)"]
        Scaler["📏 StandardScaler<br/>(Zero-mean, unit-variance transform)"]
        Classifier["🎯 Logistic Regression<br/>(Trained with max_iter=1000)"]
        
        Validator -->|4. Validated Dataframe| Imputer
        Imputer --> Scaler
        Scaler --> Classifier
    end

    subgraph Explainability["🔍 Explainable AI (XAI) Engine"]
        Probability["📈 Sigmoid Calibration<br/>P(y=1|X) Risk Tier"]
        Attribution["🧮 Linear Feature Attribution<br/>Contribution = Weight × Scaled_Value<br/>Direction: Increases (+) vs Decreases (-) Risk"]
        
        Classifier --> Probability
        Scaler --> Attribution
        Classifier --> Attribution
    end

    Probability -->|5. Risk score & tier| ResponseBuilder["📦 JSON Response Assembler"]
    Attribution -->|Top 5 sorted by magnitude| ResponseBuilder
    Validator -.->|Uncertainty warnings| ResponseBuilder

    ResponseBuilder -->|6. Unified Clinical Payload| VizEngine
    VizEngine -->|7. Actionable Medical Insights| User

    classDef client fill:#f0fdfa,stroke:#0d9488,stroke-width:2px,color:#0f766e;
    classDef gateway fill:#eff6ff,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef ml fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef xai fill:#fffbeb,stroke:#d97706,stroke-width:2px,color:#92400e;
    
    class Client client;
    class Gateway gateway;
    class MLEngine ml;
    class Explainability xai;
```

---

## 🔬 Datasets & Provenance

| Disease | Dataset | Source | Records | Features | Target |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Diabetes** | [Pima Indians Diabetes](https://www.openml.org/search?type=data&id=37) | National Institute of Diabetes and Digestive and Kidney Diseases via OpenML | 768 | 8 | Diabetes diagnosis (1 = diabetic, 0 = non-diabetic) |
| **Breast Cancer** | [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | UCI Machine Learning Repository via Scikit-learn | 569 | 10 | Tumor malignancy status (1 = malignant, 0 = benign) |

*Preprocessing note:* In the Diabetes dataset, biologically impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI were treated as missing values and imputed using training median values; in Breast Cancer, only the 10 core mean nuclear morphology attributes were extracted and the target was calibrated so 1 represents malignant.

---

## 📊 Empirical Model Results

Empirical metrics computed from direct pipeline training:

### 1. Holdout Test Performance (80/20 Stratified Split)

| Disease | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Confusion Matrix `[TN, FP / FN, TP]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diabetes** | 70.78% | 60.00% | 50.00% | 0.5455 | 0.8130 | `[82, 18 / 27, 27]` |
| **Breast Cancer** | 92.98% | 88.64% | 92.86% | 0.9070 | 0.9841 | `[67, 5 / 3, 39]` |

### 2. Repeated Stratified Cross-Validation (5 Folds × 10 Repeats = 50 Runs)

| Disease | Model Candidate | Mean Accuracy | Mean F1-Score | Mean ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Diabetes** | Baseline (Majority Class) | 65.10% ± 0.20% | 0.000 ± 0.000 | 0.500 ± 0.000 |
| | **Logistic Regression (Deployed)** | **76.70% ± 2.90%** | **0.624 ± 0.054** | **0.835 ± 0.028** |
| | Random Forest (200 trees) | 76.50% ± 2.50% | 0.641 ± 0.042 | 0.829 ± 0.027 |
| **Breast Cancer** | Baseline (Majority Class) | 62.70% ± 0.40% | 0.000 ± 0.000 | 0.500 ± 0.000 |
| | **Logistic Regression (Deployed)** | **93.70% ± 2.00%** | **0.912 ± 0.029** | **0.985 ± 0.006** |
| | Random Forest (200 trees) | 93.70% ± 2.40% | 0.915 ± 0.033 | 0.983 ± 0.009 |

*Model Note:* While Breast Cancer detection achieves high sensitivity (92.86%), Diabetes recall is modest (50.00%), reflecting the inherent non-linear overlap of clinical attributes in the Pima cohort and underscoring the necessity of secondary laboratory panels.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | High-performance asynchronous REST API with Swagger documentation |
| **Package Management** | [uv](https://docs.astral.sh/uv/) (Astral) | Blazing-fast deterministic Python environment resolution |
| **Machine Learning** | [Scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Joblib](https://joblib.readthedocs.io/) | Imputation, standardization, modeling, cross-validation & serialization |
| **Validation & Testing** | [Pydantic](https://docs.pydantic.dev/), [Pytest](https://docs.pytest.org/), [HTTPX](https://www.python-httpx.org/) | Strict schema type enforcement and automated regression testing |
| **Frontend Framework** | [React 19](https://react.dev/) + [Vite](https://vitejs.dev/) | Component-driven UI with instant HMR and development proxy |
| **Data Visualization** | [Recharts](https://recharts.org/) | Responsive SVG bar charts for Explainable AI feature attribution |
| **Icons & Typography** | [Lucide React](https://lucide.dev/), [Inter](https://fonts.google.com/specimen/Inter) | Clean iconography and accessible clinical typography |

---

## 📁 Project Directory Structure

```
healthrisk-dashboard/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── train.py
│   ├── models/
│   │   ├── breast_cancer.joblib
│   │   ├── breast_cancer_metrics.json
│   │   ├── diabetes.joblib
│   │   └── diabetes_metrics.json
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
│   ├── package-lock.json
│   ├── package.json
│   └── vite.config.js
├── docs/
│   └── screenshots/
│       ├── 01-dashboard-empty.png
│       ├── 02-diabetes-high-risk.png
│       ├── 03-diabetes-low-risk.png
│       ├── 04-breast-cancer-result.png
│       ├── 05-model-performance.png
│       └── 06-about-the-data.png
├── .gitignore
└── README.md
```

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health check returning operational status and available model list. |
| `GET` | `/api/diseases` | Retrieves supported diseases, field schemas (units, limits, defaults), and sample patient profiles. |
| `GET` | `/api/metrics/{disease}` | Returns holdout metrics, cross-validation benchmarks, and dataset summary for the given disease. |
| `POST` | `/api/predict/{disease}` | Validates physiological inputs, evaluates risk probability, and returns top 5 contributing factors with warnings. |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **uv** package manager (`curl -LsSf https://astral.sh/uv/install.sh` or `pip install uv`)
- **Node.js 18+** & **npm**

### Step 1: Start the Backend

```bash
cd backend
uv sync
uv run python -m app.train
uv run pytest -v
uv run uvicorn app.main:app --port 8000
```
Backend API will be running at `http://localhost:8000` (interactive documentation available at `http://localhost:8000/docs`).

### Step 2: Start the Frontend

Open a new terminal session:
```bash
cd frontend
npm install
npm run dev
```
Frontend application will be accessible at `http://localhost:5173`.

---

## ⚠️ Limitations & Clinical Caution

1. **Small Reference Datasets:** The Pima Diabetes ($N=768$) and Wisconsin Diagnostic Breast Cancer ($N=569$) cohorts are benchmark reference datasets that may not represent broad global populations.
2. **Not Clinically Validated:** This application is developed for educational exploration. It has not undergone formal clinical trial evaluation and must not supersede certified healthcare practitioners.
3. **Modest Diabetes Recall:** Diabetes recall on holdout testing is 50.00%, meaning borderline at-risk patients require further diagnostic screening (such as HbA1c panels).
4. **Descriptive Associations:** Model predictions describe statistical patterns identified in retrospective cohorts rather than definitive medical diagnoses.

---

## 👨‍💻 Author

<table style="border: none;">
  <tr>
    <td width="90" align="center" style="border: none;">
      <a href="https://github.com/ahmadaquib">
        <img src="https://github.com/ahmadaquib.png" width="80" height="80" style="border-radius: 50%; box-shadow: 0 4px 12px rgba(13, 148, 136, 0.25);" alt="Aquib Ahmad" />
      </a>
    </td>
    <td style="border: none;">
      <h3 style="margin-top: 0; margin-bottom: 4px;">Aquib</h3>
      <p style="margin: 0; color: #64748b; font-size: 0.95rem;">
        MLOps &amp; AI Systems Engineer &bull; Building explainable, high-reliability healthcare AI applications.
      </p>
      <p style="margin-top: 8px; margin-bottom: 0;">
        <a href="https://github.com/ahmadaquib">
          <img src="https://img.shields.io/badge/GitHub-ahmadaquib-181717?style=flat&logo=github&logoColor=white" alt="GitHub" />
        </a>
      </p>
    </td>
  </tr>
</table>
