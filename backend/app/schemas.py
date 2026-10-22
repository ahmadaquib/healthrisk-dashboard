from typing import Any, Dict, List
from pydantic import BaseModel


class FieldSchema(BaseModel):
    name: str
    label: str
    unit: str
    description: str
    training_min: float
    training_max: float
    training_mean: float
    min_sane: float
    max_sane: float
    step: float
    default: float


class DiseaseInfo(BaseModel):
    id: str
    name: str
    description: str
    fields: List[FieldSchema]
    example_patients: Dict[str, Any]


class FactorContribution(BaseModel):
    feature: str
    label: str
    contribution: float
    impact: float
    direction: str  # "increases risk" or "decreases risk"
    value: float
    unit: str


class PredictionResponse(BaseModel):
    disease: str
    probability: float
    risk_level: str  # "Low" (<33%), "Moderate" (33-66%), "High" (>66%)
    top_factors: List[FactorContribution]
    warnings: List[str]
    input_values: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    service: str
    available_models: List[str]
