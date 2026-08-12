"""Strict public API contract for Huy's final CatBoost model."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.features.build_features import MEDICATION_FEATURES, REQUEST_FEATURES


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    contract_validated: bool
    model_version: str | None = None
    feature_set: str | None = None
    message: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version: str
    risk_score: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    prediction: Literal[0, 1]
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


Race = Literal["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]
Gender = Literal["Female", "Male"]
Age = Literal[
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]
MedicationState = Literal["No", "Steady", "Up", "Down"]
MaxGlucose = Literal["None", "Unknown", "Norm", ">200", ">300"] | None
A1CResult = Literal["None", "Unknown", "Norm", ">7", ">8"] | None
DischargeDisposition = Literal[
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    27,
    28,
]

NUMERIC_BOUNDS = {
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (1, 132),
    "num_procedures": (0, 6),
    "num_medications": (1, 81),
    "number_outpatient": (0, 42),
    "number_emergency": (0, 76),
    "number_inpatient": (0, 21),
    "number_diagnoses": (1, 16),
}


def _field_definitions() -> dict[str, tuple[Any, Any]]:
    definitions: dict[str, tuple[Any, Any]] = {}
    for feature in REQUEST_FEATURES:
        if feature == "race":
            definitions[feature] = (Race, ...)
        elif feature == "gender":
            definitions[feature] = (Gender, ...)
        elif feature == "age":
            definitions[feature] = (Age, ...)
        elif feature == "admission_type_id":
            definitions[feature] = (Literal[1, 2, 3, 4, 5, 6, 7, 8], ...)
        elif feature == "discharge_disposition_id":
            definitions[feature] = (DischargeDisposition, ...)
        elif feature == "admission_source_id":
            definitions[feature] = (
                Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 20, 22, 25],
                ...,
            )
        elif feature in NUMERIC_BOUNDS:
            minimum, maximum = NUMERIC_BOUNDS[feature]
            definitions[feature] = (Annotated[int, Field(ge=minimum, le=maximum)], ...)
        elif feature == "max_glu_serum":
            definitions[feature] = (MaxGlucose, ...)
        elif feature == "A1Cresult":
            definitions[feature] = (A1CResult, ...)
        elif feature in MEDICATION_FEATURES:
            definitions[feature] = (MedicationState, ...)
        elif feature == "change":
            definitions[feature] = (Literal["No", "Ch"], ...)
        elif feature == "diabetesMed":
            definitions[feature] = (Literal["No", "Yes"], ...)
        elif feature == "diag_1":
            definitions[feature] = (
                Annotated[str, Field(pattern=r"^(?:[VE]\d+(?:\.\d+)?|\d+(?:\.\d+)?)$")],
                ...,
            )
        else:  # pragma: no cover - guarded by the frozen constant
            raise RuntimeError(f"No schema definition for Huy feature: {feature}")
    return definitions


def prediction_example() -> dict[str, Any]:
    example: dict[str, Any] = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[80-90)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 10,
        "num_lab_procedures": 70,
        "num_procedures": 2,
        "num_medications": 25,
        "number_outpatient": 3,
        "number_emergency": 2,
        "number_inpatient": 4,
        "number_diagnoses": 9,
        "max_glu_serum": ">300",
        "A1Cresult": ">8",
        "change": "Ch",
        "diabetesMed": "Yes",
        "diag_1": "250.13",
    }
    for medication in MEDICATION_FEATURES:
        example[medication] = "No"
    example["insulin"] = "Up"
    return {feature: example[feature] for feature in REQUEST_FEATURES}


PredictionRequest = create_model(
    "PredictionRequest",
    __config__=ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        title="HuyPredictionRequest",
        json_schema_extra={"example": prediction_example()},
    ),
    **_field_definitions(),
)
PredictionRequest.__doc__ = (
    "Exactly 40 raw encounter fields. The service derives and scales Huy's 52 model features."
)
