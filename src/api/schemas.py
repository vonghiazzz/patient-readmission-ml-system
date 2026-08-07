"""Versioned request and response contracts for the readmission API."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmptyString = Annotated[str, Field(min_length=1, max_length=100)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]


class HealthResponse(BaseModel):
    """Response returned by the process health endpoint."""

    status: str = Field(examples=["healthy"])
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Response describing whether model-backed inference is available."""

    status: str = Field(examples=["ready"])
    mode: str = Field(examples=["mock"])
    model_loaded: bool
    message: str


class PredictionRequest(BaseModel):
    """Leakage-safe patient features available at prediction time.

    Identifiers, source/derived targets and engineered aggregate features are
    deliberately excluded. The two aggregate features are computed by the
    preprocessing pipeline from the request's service and activity counts.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "race": "Caucasian",
                    "gender": "Female",
                    "age": "[60-70)",
                    "time_in_hospital": 4,
                    "num_lab_procedures": 42,
                    "num_procedures": 1,
                    "num_medications": 12,
                    "number_outpatient": 2,
                    "number_emergency": 0,
                    "number_inpatient": 1,
                    "number_diagnoses": 7,
                    "insulin": "Steady",
                    "change": "No",
                    "diabetesmed": "Yes",
                }
            ]
        },
    )

    race: NonEmptyString | None = Field(default=None, examples=["Caucasian"])
    gender: NonEmptyString = Field(examples=["Female"])
    age: NonEmptyString = Field(examples=["[60-70)"])
    weight: NonEmptyString | None = Field(default=None, examples=["[75-100)"])

    admission_type_id: NonNegativeInt | None = None
    discharge_disposition_id: NonNegativeInt | None = None
    admission_source_id: NonNegativeInt | None = None
    time_in_hospital: PositiveInt

    payer_code: NonEmptyString | None = None
    medical_specialty: NonEmptyString | None = None

    num_lab_procedures: NonNegativeInt
    num_procedures: NonNegativeInt
    num_medications: NonNegativeInt
    number_outpatient: NonNegativeInt
    number_emergency: NonNegativeInt
    number_inpatient: NonNegativeInt

    diag_1: NonEmptyString | None = None
    diag_2: NonEmptyString | None = None
    diag_3: NonEmptyString | None = None
    number_diagnoses: NonNegativeInt

    max_glu_serum: NonEmptyString | None = None
    a1cresult: NonEmptyString | None = None

    metformin: NonEmptyString | None = None
    repaglinide: NonEmptyString | None = None
    nateglinide: NonEmptyString | None = None
    chlorpropamide: NonEmptyString | None = None
    glimepiride: NonEmptyString | None = None
    acetohexamide: NonEmptyString | None = None
    glipizide: NonEmptyString | None = None
    glyburide: NonEmptyString | None = None
    tolbutamide: NonEmptyString | None = None
    pioglitazone: NonEmptyString | None = None
    rosiglitazone: NonEmptyString | None = None
    acarbose: NonEmptyString | None = None
    miglitol: NonEmptyString | None = None
    troglitazone: NonEmptyString | None = None
    tolazamide: NonEmptyString | None = None
    examide: NonEmptyString | None = None
    citoglipton: NonEmptyString | None = None
    insulin: NonEmptyString | None = None
    glyburide_metformin: NonEmptyString | None = None
    glipizide_metformin: NonEmptyString | None = None
    glimepiride_pioglitazone: NonEmptyString | None = None
    metformin_rosiglitazone: NonEmptyString | None = None
    metformin_pioglitazone: NonEmptyString | None = None
    change: NonEmptyString | None = None
    diabetesmed: NonEmptyString | None = None


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "readmission_probability": 0.0,
                    "predicted_readmission": False,
                    "risk_band": "mock",
                    "threshold": 0.5,
                    "model_version": "mock-phase-2",
                    "is_mock": True,
                }
            ]
        }
    )

    readmission_probability: float = Field(ge=0, le=1)
    predicted_readmission: bool
    risk_band: str
    threshold: float = Field(ge=0, le=1)
    model_version: str
    is_mock: bool


class ErrorDetail(BaseModel):
    """One privacy-safe input validation error."""

    location: list[str | int]
    message: str
    error_type: str


class APIError(BaseModel):
    """Stable error envelope content."""

    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Stable error response returned by API exception handlers."""

    error: APIError
