"""Draft API schemas for the Phase 1 FastAPI skeleton.

The prediction request is provisional.

Member A must later confirm:
- input field names
- data types
- categorical values
- missing-value rules
- feature-engineering requirements

Member B must later confirm:
- model feature order
- prediction output
- threshold
- risk-band rules
- explainability output
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response returned by the process health endpoint."""

    status: str = Field(examples=["healthy"])
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Phase 1 readiness response."""

    status: str = Field(examples=["ready"])
    mode: str = Field(examples=["mock"])
    model_loaded: bool
    message: str


class PredictionRequest(BaseModel):
    """Provisional request used by the Phase 1 mock endpoint."""

    age: str = Field(examples=["[60-70)"])
    gender: str = Field(examples=["Female"])

    number_inpatient: int = Field(
        ge=0,
        examples=[1],
    )
    number_emergency: int = Field(
        ge=0,
        examples=[0],
    )
    number_outpatient: int = Field(
        ge=0,
        examples=[2],
    )


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    readmission_probability: float = Field(
        ge=0,
        le=1,
    )
    predicted_readmission: bool
    risk_band: str

    threshold: float = Field(
        ge=0,
        le=1,
    )

    model_version: str
    is_mock: bool
