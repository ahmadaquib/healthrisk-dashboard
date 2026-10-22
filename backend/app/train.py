import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


DIABETES_METADATA = {
    "Pregnancies": {
        "label": "Pregnancies",
        "unit": "count",
        "description": "Number of times pregnant",
        "min_sane": 0,
        "max_sane": 25,
        "step": 1,
    },
    "Glucose": {
        "label": "Plasma Glucose",
        "unit": "mg/dL",
        "description": "2-hour oral glucose tolerance test",
        "min_sane": 40,
        "max_sane": 400,
        "step": 1,
    },
    "BloodPressure": {
        "label": "Diastolic Blood Pressure",
        "unit": "mm Hg",
        "description": "Resting blood pressure",
        "min_sane": 30,
        "max_sane": 200,
        "step": 1,
    },
    "SkinThickness": {
        "label": "Triceps Skinfold Thickness",
        "unit": "mm",
        "description": "Measurement of body fat layer",
        "min_sane": 2,
        "max_sane": 100,
        "step": 1,
    },
    "Insulin": {
        "label": "2-Hour Serum Insulin",
        "unit": "mu U/ml",
        "description": "Fasting or 2-hour serum insulin",
        "min_sane": 5,
        "max_sane": 900,
        "step": 1,
    },
    "BMI": {
        "label": "Body Mass Index (BMI)",
        "unit": "kg/m²",
        "description": "Weight in kg / (height in m)²",
        "min_sane": 10.0,
        "max_sane": 70.0,
        "step": 0.1,
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Pedigree Function",
        "unit": "score",
        "description": "Genetic risk score based on family history",
        "min_sane": 0.05,
        "max_sane": 3.0,
        "step": 0.01,
    },
    "Age": {
        "label": "Age",
        "unit": "years",
        "description": "Patient age in years",
        "min_sane": 18,
        "max_sane": 120,
        "step": 1,
    },
}

BREAST_CANCER_METADATA = {
    "mean_radius": {
        "label": "Mean Radius",
        "unit": "mm",
        "description": "Mean distance from center to points on the perimeter",
        "min_sane": 5.0,
        "max_sane": 35.0,
        "step": 0.01,
    },
    "mean_texture": {
        "label": "Mean Texture",
        "unit": "std dev",
        "description": "Standard deviation of gray-scale values",
        "min_sane": 5.0,
        "max_sane": 45.0,
        "step": 0.01,
    },
    "mean_perimeter": {
        "label": "Mean Perimeter",
        "unit": "mm",
        "description": "Mean size of the core tumor perimeter",
        "min_sane": 30.0,
        "max_sane": 250.0,
        "step": 0.1,
    },
    "mean_area": {
        "label": "Mean Area",
        "unit": "mm²",
        "description": "Mean tumor area",
        "min_sane": 100.0,
        "max_sane": 3000.0,
        "step": 0.1,
    },
    "mean_smoothness": {
        "label": "Mean Smoothness",
        "unit": "ratio",
        "description": "Local variation in radius lengths",
        "min_sane": 0.04,
        "max_sane": 0.25,
        "step": 0.001,
    },
    "mean_compactness": {
        "label": "Mean Compactness",
        "unit": "ratio",
        "description": "Perimeter² / area - 1.0",
        "min_sane": 0.01,
        "max_sane": 0.40,
        "step": 0.001,
    },
    "mean_concavity": {
        "label": "Mean Concavity",
        "unit": "ratio",
        "description": "Severity of concave portions of the contour",
        "min_sane": 0.0,
        "max_sane": 0.50,
        "step": 0.001,
    },
    "mean_concave_points": {
        "label": "Mean Concave Points",
        "unit": "ratio",
        "description": "Number of concave portions of the contour",
        "min_sane": 0.0,
        "max_sane": 0.25,
        "step": 0.001,
    },
    "mean_symmetry": {
        "label": "Mean Symmetry",
        "unit": "ratio",
        "description": "Nuclear symmetry score",
        "min_sane": 0.10,
        "max_sane": 0.35,
        "step": 0.001,
    },
    "mean_fractal_dimension": {
        "label": "Mean Fractal Dimension",
        "unit": "ratio",
        "description": "Coastline approximation - 1",
        "min_sane": 0.03,
        "max_sane": 0.12,
        "step": 0.001,
    },
}


def load_diabetes_data():
    try:
        print("Fetching diabetes dataset from OpenML...")
        data = fetch_openml("diabetes", version=1, as_frame=True)
        df = data.frame.copy()
        col_map = {
            "preg": "Pregnancies",
            "plas": "Glucose",
            "pres": "BloodPressure",
            "skin": "SkinThickness",
            "insu": "Insulin",
            "mass": "BMI",
            "pedi": "DiabetesPedigreeFunction",
            "age": "Age",
            "class": "Outcome",
        }
        df = df.rename(columns=col_map)
        if df["Outcome"].dtype == object or str(df["Outcome"].dtype) == "category":
            df["Outcome"] = df["Outcome"].map({"tested_positive": 1, "tested_negative": 0})
    except Exception as e:
        print(f"fetch_openml failed ({e}), falling back to GitHub mirror...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        names = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Outcome",
        ]
        df = pd.read_csv(url, names=names)

    # In Glucose, BloodPressure, SkinThickness, Insulin and BMI, treat 0 as missing (NaN)
    zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_nan:
        df[col] = df[col].replace(0, np.nan)

    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    X = df[features]
    y = df["Outcome"].astype(int)
    return X, y, DIABETES_METADATA


def load_breast_cancer_data():
    print("Loading breast cancer dataset from scikit-learn...")
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    # Use ONLY the 10 "mean ..." features
    raw_mean_features = [c for c in df.columns if c.startswith("mean ")]
    X = df[raw_mean_features].copy()

    # Standardize column names to snake_case
    col_map = {c: c.replace(" ", "_") for c in raw_mean_features}
    X = X.rename(columns=col_map)

    # In sklearn breast cancer: 0 = malignant, 1 = benign.
    # Instruction: "Flip the target so 1 = malignant."
    y = (1 - df["target"]).astype(int)

    return X, y, BREAST_CANCER_METADATA


def train_disease_model(disease_name, X, y, feature_metadata):
    print(f"\n==================================================")
    print(f"Training Model for: {disease_name.upper()}")
    print(f"Total samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Class distribution:\n{y.value_counts().to_dict()}")

    # 80/20 stratified train/test split, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Pipeline: SimpleImputer(median) -> StandardScaler -> LogisticRegression(max_iter=1000)
    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    lr_pipeline.fit(X_train, y_train)

    # Predictions on holdout test set
    y_pred = lr_pipeline.predict(X_test)
    y_prob = lr_pipeline.predict_proba(X_test)[:, 1]

    # Metrics on holdout test set
    holdout_metrics = {
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
        "f1": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, y_prob), 4)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # RepeatedStratifiedKFold (5 folds x 10 repeats, random_state=42)
    print("Running RepeatedStratifiedKFold (5 folds x 10 repeats)...")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    scoring = ["accuracy", "f1", "roc_auc"]

    # 1. Dummy Baseline (majority class)
    dummy_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ])
    dummy_cv = cross_validate(dummy_pipe, X, y, cv=rskf, scoring=scoring, n_jobs=-1)

    # 2. Logistic Regression
    lr_cv_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr_cv = cross_validate(lr_cv_pipe, X, y, cv=rskf, scoring=scoring, n_jobs=-1)

    # 3. Random Forest (200 trees, random_state=42)
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    rf_cv = cross_validate(rf_pipe, X, y, cv=rskf, scoring=scoring, n_jobs=-1)

    cv_comparison = {
        "Baseline (Majority Class)": {
            "accuracy_mean": float(round(np.mean(dummy_cv["test_accuracy"]), 4)),
            "accuracy_std": float(round(np.std(dummy_cv["test_accuracy"]), 4)),
            "f1_mean": float(round(np.mean(dummy_cv["test_f1"]), 4)),
            "f1_std": float(round(np.std(dummy_cv["test_f1"]), 4)),
            "roc_auc_mean": float(round(np.mean(dummy_cv["test_roc_auc"]), 4)),
            "roc_auc_std": float(round(np.std(dummy_cv["test_roc_auc"]), 4)),
        },
        "Logistic Regression": {
            "accuracy_mean": float(round(np.mean(lr_cv["test_accuracy"]), 4)),
            "accuracy_std": float(round(np.std(lr_cv["test_accuracy"]), 4)),
            "f1_mean": float(round(np.mean(lr_cv["test_f1"]), 4)),
            "f1_std": float(round(np.std(lr_cv["test_f1"]), 4)),
            "roc_auc_mean": float(round(np.mean(lr_cv["test_roc_auc"]), 4)),
            "roc_auc_std": float(round(np.std(lr_cv["test_roc_auc"]), 4)),
        },
        "Random Forest (200 trees)": {
            "accuracy_mean": float(round(np.mean(rf_cv["test_accuracy"]), 4)),
            "accuracy_std": float(round(np.std(rf_cv["test_accuracy"]), 4)),
            "f1_mean": float(round(np.mean(rf_cv["test_f1"]), 4)),
            "f1_std": float(round(np.std(rf_cv["test_f1"]), 4)),
            "roc_auc_mean": float(round(np.mean(rf_cv["test_roc_auc"]), 4)),
            "roc_auc_std": float(round(np.std(rf_cv["test_roc_auc"]), 4)),
        },
    }

    # Per-feature min/max/mean from training data (ignoring NaNs)
    features_info = []
    for col in X.columns:
        meta = feature_metadata.get(col, {})
        col_min = float(round(X_train[col].min(), 4))
        col_max = float(round(X_train[col].max(), 4))
        col_mean = float(round(X_train[col].mean(), 4))
        features_info.append({
            "name": col,
            "label": meta.get("label", col),
            "unit": meta.get("unit", ""),
            "description": meta.get("description", ""),
            "training_min": col_min,
            "training_max": col_max,
            "training_mean": col_mean,
            "min_sane": meta.get("min_sane", col_min),
            "max_sane": meta.get("max_sane", col_max),
            "step": meta.get("step", 0.01 if isinstance(col_mean, float) else 1),
            "default": col_mean,
        })

    # Pick example patients from test set: highest and lowest predicted probability
    high_idx = int(np.argmax(y_prob))
    low_idx = int(np.argmin(y_prob))

    # Convert test row to dictionary with clean python types
    high_risk_row = X_test.iloc[high_idx].to_dict()
    low_risk_row = X_test.iloc[low_idx].to_dict()

    # Fill any NaNs in example rows with median from training
    for k, v in high_risk_row.items():
        if pd.isna(v):
            high_risk_row[k] = float(round(X_train[k].median(), 2))
        else:
            high_risk_row[k] = float(round(v, 4))

    for k, v in low_risk_row.items():
        if pd.isna(v):
            low_risk_row[k] = float(round(X_train[k].median(), 2))
        else:
            low_risk_row[k] = float(round(v, 4))

    example_patients = {
        "high_risk": {
            "values": high_risk_row,
            "probability": float(round(y_prob[high_idx], 4)),
            "actual_label": int(y_test.iloc[high_idx]),
        },
        "low_risk": {
            "values": low_risk_row,
            "probability": float(round(y_prob[low_idx], 4)),
            "actual_label": int(y_test.iloc[low_idx]),
        },
    }

    # Model coefficients and intercept
    model_coefs = dict(zip(X.columns, lr_pipeline.named_steps["classifier"].coef_[0]))
    intercept = float(lr_pipeline.named_steps["classifier"].intercept_[0])

    # Save final pipeline
    model_path = MODELS_DIR / f"{disease_name}.joblib"
    joblib.dump(lr_pipeline, model_path)
    print(f"Saved model pipeline to: {model_path}")

    # Dataset Info
    if disease_name == "diabetes":
        dataset_info = {
            "name": "Pima Indians Diabetes Dataset",
            "source": "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)",
            "source_details": "Loaded via OpenML 'diabetes' v1 (ID 37)",
            "source_url": "https://www.openml.org/search?type=data&id=37",
            "total_records": len(X),
            "num_features": X.shape[1],
            "feature_names": list(X.columns),
            "target_definition": "Diabetes diagnosis outcome (1 = diabetic, 0 = non-diabetic)",
            "class_balance": {
                "Non-diabetic (0)": int((y == 0).sum()),
                "Diabetic (1)": int((y == 1).sum()),
            },
            "class_balance_percent": {
                "Non-diabetic (0)": f"{((y == 0).sum() / len(y) * 100):.1f}%",
                "Diabetic (1)": f"{((y == 1).sum() / len(y) * 100):.1f}%",
            },
            "preprocessing_summary": "Zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI treated as missing (NaN) and median-imputed.",
        }
    else:
        dataset_info = {
            "name": "Breast Cancer Wisconsin (Diagnostic) Dataset",
            "source": "UCI Machine Learning Repository",
            "source_details": "Loaded from scikit-learn (load_breast_cancer)",
            "source_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
            "total_records": len(X),
            "num_features": X.shape[1],
            "feature_names": list(X.columns),
            "target_definition": "Tumor malignancy status (1 = malignant, 0 = benign)",
            "class_balance": {
                "Benign (0)": int((y == 0).sum()),
                "Malignant (1)": int((y == 1).sum()),
            },
            "class_balance_percent": {
                "Benign (0)": f"{((y == 0).sum() / len(y) * 100):.1f}%",
                "Malignant (1)": f"{((y == 1).sum() / len(y) * 100):.1f}%",
            },
            "preprocessing_summary": "Target calibrated so 1 = malignant; strictly restricted to the 10 core mean nuclear morphology features.",
        }

    # Metrics JSON
    full_metrics = {
        "disease": disease_name,
        "dataset_info": dataset_info,
        "features": features_info,
        "holdout_metrics": holdout_metrics,
        "cv_comparison": cv_comparison,
        "example_patients": example_patients,
        "coefficients": {k: float(round(v, 4)) for k, v in model_coefs.items()},
        "intercept": round(intercept, 4),
        "test_size": len(X_test),
        "train_size": len(X_train),
    }

    metrics_path = MODELS_DIR / f"{disease_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"Saved metrics JSON to: {metrics_path}")

    return full_metrics


def print_summary(diabetes_res, cancer_res):
    print("\n" + "=" * 80)
    print("                         MODEL TRAINING SUMMARY                         ")
    print("=" * 80)

    for res in [diabetes_res, cancer_res]:
        d_name = res["disease"].upper()
        hm = res["holdout_metrics"]
        cv = res["cv_comparison"]

        print(f"\n--- {d_name} (Holdout Test 80/20 Stratified Split) ---")
        print(f"  Accuracy:  {hm['accuracy'] * 100:.2f}%")
        print(f"  Precision: {hm['precision'] * 100:.2f}%")
        print(f"  Recall:    {hm['recall'] * 100:.2f}%")
        print(f"  F1 Score:  {hm['f1']:.4f}")
        print(f"  ROC-AUC:   {hm['roc_auc']:.4f}")
        cm = hm["confusion_matrix"]
        print(f"  Confusion Matrix: [TN: {cm[0][0]}, FP: {cm[0][1]} | FN: {cm[1][0]}, TP: {cm[1][1]}]")

        print(f"\n  Repeated Stratified 5-Fold x 10 Repeats Cross-Validation:")
        print(f"  {'Model':<28} | {'Accuracy':<14} | {'F1-Score':<14} | {'ROC-AUC':<14}")
        print("  " + "-" * 74)
        for m_name, m_vals in cv.items():
            acc_str = f"{m_vals['accuracy_mean']:.3f} +/- {m_vals['accuracy_std']:.3f}"
            f1_str = f"{m_vals['f1_mean']:.3f} +/- {m_vals['f1_std']:.3f}"
            auc_str = f"{m_vals['roc_auc_mean']:.3f} +/- {m_vals['roc_auc_std']:.3f}"
            print(f"  {m_name:<28} | {acc_str:<14} | {f1_str:<14} | {auc_str:<14}")

        ex = res["example_patients"]
        print(f"\n  Example Patients Test Set Predictions:")
        print(f"  - High Risk: Prob = {ex['high_risk']['probability']*100:.1f}%, Actual = {ex['high_risk']['actual_label']}")
        print(f"  - Low Risk:  Prob = {ex['low_risk']['probability']*100:.1f}%, Actual = {ex['low_risk']['actual_label']}")

    print("\n" + "=" * 80)
    print("Training finished successfully. Models & metrics ready.")
    print("=" * 80 + "\n")


def main():
    # 1. Diabetes
    X_diab, y_diab, meta_diab = load_diabetes_data()
    diab_metrics = train_disease_model("diabetes", X_diab, y_diab, meta_diab)

    # 2. Breast Cancer
    X_cancer, y_cancer, meta_cancer = load_breast_cancer_data()
    cancer_metrics = train_disease_model("breast_cancer", X_cancer, y_cancer, meta_cancer)

    # 3. Print clean summary table
    print_summary(diab_metrics, cancer_metrics)


if __name__ == "__main__":
    main()
