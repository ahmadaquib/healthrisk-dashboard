import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    DiseaseInfo,
    FactorContribution,
    FieldSchema,
    HealthResponse,
    PredictionResponse,
)

from contextlib import asynccontextmanager

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# In-memory cache for pipelines and metrics
PIPELINES = {}
METRICS = {}


def normalize_disease_id(disease: str) -> str:
    cleaned = disease.lower().replace("-", "_").strip()
    if cleaned in ["diabetes", "diab"]:
        return "diabetes"
    if cleaned in ["breast_cancer", "cancer", "breastcancer"]:
        return "breast_cancer"
    return cleaned


def load_disease_artifacts(disease_id: str):
    if disease_id not in PIPELINES:
        model_file = MODELS_DIR / f"{disease_id}.joblib"
        metrics_file = MODELS_DIR / f"{disease_id}_metrics.json"

        if not model_file.exists() or not metrics_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model artifacts for disease '{disease_id}' not found. Please run training script first.",
            )

        PIPELINES[disease_id] = joblib.load(model_file)
        with open(metrics_file, "r") as f:
            METRICS[disease_id] = json.load(f)

    return PIPELINES[disease_id], METRICS[disease_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload models on startup
    for d in ["diabetes", "breast_cancer"]:
        try:
            load_disease_artifacts(d)
        except Exception:
            pass
    yield

app = FastAPI(
    title="AI Disease Risk Prediction API",
    description="College Mini-Project - AI for Healthcare Lab (Experiment 10)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Vite frontend (http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    available = []
    for d in ["diabetes", "breast_cancer"]:
        if (MODELS_DIR / f"{d}.joblib").exists():
            available.append(d)
    return HealthResponse(
        status="healthy",
        service="AI Disease Risk Prediction API",
        available_models=available,
    )


@app.get("/api/diseases", response_model=List[DiseaseInfo])
def get_diseases():
    diseases_meta = [
        {
            "id": "diabetes",
            "name": "Diabetes Risk",
            "description": "Predicts the likelihood of Type 2 Diabetes based on clinical metabolic factors.",
        },
        {
            "id": "breast_cancer",
            "name": "Breast Cancer Risk",
            "description": "Predicts whether breast cell mass characteristics indicate malignant or benign status.",
        },
    ]

    result = []
    for d in diseases_meta:
        disease_id = d["id"]
        _, metrics = load_disease_artifacts(disease_id)

        field_schemas = [FieldSchema(**feat) for feat in metrics["features"]]

        result.append(
            DiseaseInfo(
                id=disease_id,
                name=d["name"],
                description=d["description"],
                fields=field_schemas,
                example_patients=metrics["example_patients"],
            )
        )

    return result


@app.get("/api/metrics/{disease}")
def get_metrics(disease: str):
    disease_id = normalize_disease_id(disease)
    if disease_id not in ["diabetes", "breast_cancer"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown disease: '{disease}'. Valid options are 'diabetes' or 'breast_cancer'.",
        )
    _, metrics = load_disease_artifacts(disease_id)
    return metrics


@app.post("/api/predict/{disease}", response_model=PredictionResponse)
def predict_risk(disease: str, payload: Dict[str, Any]):
    disease_id = normalize_disease_id(disease)
    if disease_id not in ["diabetes", "breast_cancer"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown disease: '{disease}'. Supported diseases are 'diabetes' and 'breast_cancer'.",
        )

    pipeline, metrics = load_disease_artifacts(disease_id)
    features_info = {f["name"]: f for f in metrics["features"]}
    expected_fields = [f["name"] for f in metrics["features"]]

    # Check for missing fields
    missing_fields = []
    for f in expected_fields:
        # Check standard key or key with space/underscore variations
        if f not in payload:
            alt_key = f.replace("_", " ") if "_" in f else f.replace(" ", "_")
            if alt_key in payload:
                payload[f] = payload[alt_key]
            else:
                missing_fields.append(f)

    if missing_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required fields: {', '.join(missing_fields)}",
        )

    # Validate numeric types and sane physiological limits
    validated_values = {}
    validation_errors = []
    warnings = []

    for f_name in expected_fields:
        val = payload[f_name]
        try:
            num_val = float(val)
        except (ValueError, TypeError):
            validation_errors.append(f"Field '{f_name}' must be a valid number, got '{val}'")
            continue

        meta = features_info[f_name]
        min_sane = meta.get("min_sane")
        max_sane = meta.get("max_sane")

        # Impossible value check (reject clearly impossible values like negative age or glucose 5000)
        if min_sane is not None and num_val < min_sane:
            validation_errors.append(
                f"Value {num_val} for '{meta['label']}' is below biologically possible limit of {min_sane} {meta.get('unit', '')}"
            )
        elif max_sane is not None and num_val > max_sane:
            validation_errors.append(
                f"Value {num_val} for '{meta['label']}' exceeds biologically possible limit of {max_sane} {meta.get('unit', '')}"
            )

        # Plausible but out-of-training-range warning check
        t_min = meta["training_min"]
        t_max = meta["training_max"]
        if num_val < t_min:
            warnings.append(
                f"{meta['label']} ({num_val} {meta.get('unit', '')}) is below training set minimum ({t_min} {meta.get('unit', '')}). Model prediction may have increased uncertainty."
            )
        elif num_val > t_max:
            warnings.append(
                f"{meta['label']} ({num_val} {meta.get('unit', '')}) exceeds training set maximum ({t_max} {meta.get('unit', '')}). Model prediction may have increased uncertainty."
            )

        validated_values[f_name] = num_val

    if validation_errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="; ".join(validation_errors),
        )

    # Prepare DataFrame matching training feature columns
    df_input = pd.DataFrame([validated_values])[expected_fields]

    # Predict probability
    prob = float(pipeline.predict_proba(df_input)[0, 1])

    # Determine risk level: Low (<33%), Moderate (33-66%), High (>66%)
    if prob < 0.33:
        risk_level = "Low"
    elif prob <= 0.66:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Compute feature contributions (top factors)
    # Pipeline: SimpleImputer -> StandardScaler -> LogisticRegression
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["classifier"]

    imputed = imputer.transform(df_input)
    scaled = scaler.transform(imputed)[0]
    coefs = clf.coef_[0]

    factors = []
    for i, col in enumerate(expected_fields):
        contribution = float(coefs[i] * scaled[i])
        direction = "increases risk" if contribution > 0 else "decreases risk"
        meta = features_info[col]
        factors.append(
            FactorContribution(
                feature=col,
                label=meta["label"],
                contribution=float(round(contribution, 4)),
                impact=float(round(abs(contribution), 4)),
                direction=direction,
                value=float(round(validated_values[col], 4)),
                unit=meta.get("unit", ""),
            )
        )

    # Top 5 factors sorted by absolute contribution magnitude
    top_factors = sorted(factors, key=lambda x: x.impact, reverse=True)[:5]

    return PredictionResponse(
        disease=disease_id,
        probability=float(round(prob, 4)),
        risk_level=risk_level,
        top_factors=top_factors,
        warnings=warnings,
        input_values=validated_values,
    )
