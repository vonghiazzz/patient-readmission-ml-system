from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_settings_dependency
from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)
from src.config.settings import Settings

router = APIRouter()

SettingsDependency = Annotated[
    Settings,
    Depends(get_settings_dependency),
]


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"],
)
def health(
    settings: SettingsDependency,
) -> HealthResponse:
    """Confirm that the FastAPI process is running."""

    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["Operations"],
)
def readiness() -> ReadinessResponse:
    """Return the Phase 2 mock readiness state."""

    return ReadinessResponse(
        status="ready",
        mode="mock",
        model_loaded=False,
        message=("Phase 2 API contract is ready; real model artifacts are not integrated yet."),
    )


@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "The request does not satisfy the API contract.",
        },
        500: {
            "model": ErrorResponse,
            "description": "An unexpected server error occurred.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Required model artifacts are unavailable.",
        },
    },
    tags=["Prediction"],
    summary="Return a Phase 2 mock prediction",
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """Validate the request and return an explicit mock response."""

    _ = request

    return PredictionResponse(
        readmission_probability=0.0,
        predicted_readmission=False,
        risk_band="mock",
        threshold=0.5,
        model_version="mock-phase-2",
        is_mock=True,
    )
