"""Manifest-driven request and response contracts for the production API."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.api.catboost_dependencies import (
    CATBOOST_CATEGORICAL_FEATURES,
    CATBOOST_FEATURES,
)
from src.api.dependencies import FROZEN_SCHEMA_ARTIFACT_DIR, load_schema_contract


class HealthResponse(BaseModel):
    status: str = Field(examples=["healthy"])
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: str = Field(examples=["ready"])
    model_loaded: bool
    contract_validated: bool
    model_version: str | None = None
    feature_set: str | None = None
    message: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "model_version": "1.0.0",
                "risk_score": 0.23,
                "decision_threshold": 0.17,
                "prediction": 1,
                "status": "high_risk",
            }
        },
    )

    model_version: str
    risk_score: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    prediction: int = Field(ge=0, le=1)
    status: str = Field(examples=["high_risk", "not_high_risk"])


class CatBoostPredictionResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "model_type": "CatBoostClassifier",
                "artifact_sha256": (
                    "4d5c1217e3f07d976d16b98d946639a742c26e4137066f7e6853b9c67cd05162"
                ),
                "risk_score": 0.31,
                "decision_threshold": 0.5,
                "prediction": 0,
                "status": "not_high_risk",
            }
        },
    )

    model_type: Literal["CatBoostClassifier"]
    artifact_sha256: str = Field(min_length=64, max_length=64)
    risk_score: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    prediction: int = Field(ge=0, le=1)
    status: Literal["high_risk", "not_high_risk"]


class ErrorDetail(BaseModel):
    location: list[str | int]
    message: str
    error_type: str


class APIError(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: APIError


def _prediction_field_definitions() -> dict[str, tuple[Any, Any]]:
    """Derive public field names and type semantics from frozen artifacts."""

    manifest, preprocessor = load_schema_contract(FROZEN_SCHEMA_ARTIFACT_DIR)
    request_features = manifest.get("request_features")
    if not isinstance(request_features, list) or len(request_features) != 42:
        raise RuntimeError("Production request manifest must contain exactly 42 features")

    transformer_columns = {
        name: set(columns)
        for name, _transformer, columns in preprocessor.transformers_
        if name != "remainder" and not isinstance(columns, str)
    }
    numeric = transformer_columns.get("numeric", set())
    categorical = transformer_columns.get("categorical", set())

    definitions: dict[str, tuple[Any, Any]] = {}
    for feature in request_features:
        if feature in numeric:
            minimum = 1 if feature == "time_in_hospital" else 0
            definitions[feature] = (Annotated[int, Field(ge=minimum)], ...)
        elif feature in categorical and feature.endswith("_id"):
            # These integer IDs are intentionally routed through the fitted
            # categorical pipeline; accepting int preserves training semantics.
            definitions[feature] = (Annotated[int, Field(ge=0)], ...)
        elif feature in categorical:
            definitions[feature] = (
                Annotated[str | None, Field(min_length=1, max_length=100)],
                ...,
            )
        else:
            raise RuntimeError(f"Request feature is absent from fitted preprocessor: {feature}")
    return definitions


def _synthetic_prediction_example() -> dict[str, Any]:
    """Build a non-patient example from the fitted schema vocabulary."""

    manifest, preprocessor = load_schema_contract(FROZEN_SCHEMA_ARTIFACT_DIR)
    preferred_categories = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "payer_code": "MC",
        "medical_specialty": "InternalMedicine",
        "max_glu_serum": "Missing",
        "A1Cresult": ">8",
        "insulin": "Steady",
        "diabetesMed": "Yes",
        "admission_type_id": 1,
        "admission_source_id": 7,
    }
    example: dict[str, Any] = {}
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder" or isinstance(columns, str):
            continue
        if transformer_name == "numeric":
            for column in columns:
                if column in manifest["request_features"]:
                    example[column] = 4 if column == "time_in_hospital" else 1
            continue

        encoder = transformer.named_steps["encoder"]
        for column, categories in zip(columns, encoder.categories_, strict=True):
            if column not in manifest["request_features"]:
                continue
            preferred = preferred_categories.get(column)
            available = [
                value.item() if hasattr(value, "item") else value
                for value in categories
                if str(value) != "nan"
            ]
            example[column] = preferred if preferred in available else available[0]
    return {name: example[name] for name in manifest["request_features"]}


PredictionRequest = create_model(
    "PredictionRequest",
    __config__=ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        title="PredictionRequest",
        json_schema_extra={"example": _synthetic_prediction_example()},
    ),
    **_prediction_field_definitions(),
)
PredictionRequest.__doc__ = (
    "Exactly 42 source features from the frozen V1 feature manifest. "
    "Identifiers, targets, excluded diagnoses, and derived fields are forbidden."
)


CATBOOST_CONTINUOUS_FEATURES = {
    "number_inpatient_log1p",
    "number_outpatient_log1p",
    "number_emergency_log1p",
    "service_utilization_log1p",
    "time_in_hospital|num_lab_procedures",
    "num_medications|num_lab_procedures",
    "num_medications|number_diagnoses",
    "age|number_diagnoses",
    "change|num_medications",
    "number_diagnoses|time_in_hospital",
    "num_medications|time_in_hospital_log",
    "num_medications|num_procedures_log1p",
    "num_medications|numchange_log1p",
}


def _catboost_field_definitions() -> dict[str, tuple[Any, Any]]:
    categorical = set(CATBOOST_CATEGORICAL_FEATURES)
    definitions: dict[str, tuple[Any, Any]] = {}
    for feature in CATBOOST_FEATURES:
        if feature in categorical:
            definitions[feature] = (
                Annotated[str, Field(min_length=1, max_length=100)],
                ...,
            )
        elif feature in CATBOOST_CONTINUOUS_FEATURES:
            definitions[feature] = (
                Annotated[float, Field(ge=0, allow_inf_nan=False)],
                ...,
            )
        else:
            definitions[feature] = (Annotated[int, Field(ge=0)], ...)
    return definitions


def _synthetic_catboost_example() -> dict[str, Any]:
    example: dict[str, Any] = {feature: 0 for feature in CATBOOST_FEATURES}
    example.update(
        {
            "race": "Caucasian",
            "gender": 0,
            "age": 6,
            "admission_type_id": "1",
            "discharge_disposition_id": "1",
            "admission_source_id": "7",
            "time_in_hospital": 4,
            "num_lab_procedures": 42,
            "num_procedures": 1,
            "num_medications": 12,
            "number_diagnoses": 7,
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "insulin": 1,
            "diabetesMed": 1,
            "numchange": 1,
            "level1_diag1": "1.0",
            "nummed": 2,
            "number_inpatient_log1p": 0.693147,
            "number_outpatient_log1p": 1.098612,
            "number_emergency_log1p": 0.0,
            "service_utilization_log1p": 1.386294,
            "time_in_hospital|num_lab_procedures": 168.0,
            "num_medications|num_lab_procedures": 504.0,
            "num_medications|number_diagnoses": 84.0,
            "age|number_diagnoses": 42.0,
            "change|num_medications": 0.0,
            "number_diagnoses|time_in_hospital": 28.0,
            "num_medications|time_in_hospital_log": 3.89182,
            "num_medications|num_procedures_log1p": 2.564949,
            "num_medications|numchange_log1p": 2.564949,
        }
    )
    return {feature: example[feature] for feature in CATBOOST_FEATURES}


CatBoostPredictionRequest = create_model(
    "CatBoostPredictionRequest",
    __config__=ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        title="CatBoostPredictionRequest",
        json_schema_extra={"example": _synthetic_catboost_example()},
    ),
    **_catboost_field_definitions(),
)
CatBoostPredictionRequest.__doc__ = (
    "Exactly 52 already-engineered features embedded in the separately supplied "
    "CatBoost artifact. This experimental contract is independent of XGBoost V1."
)
