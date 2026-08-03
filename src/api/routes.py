from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_settings_dependency
from src.api.schemas import (
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
    """Return the Phase 1 mock readiness state."""

    return ReadinessResponse(
        status="ready",
        mode="mock",
        model_loaded=False,
        message=("Phase 1 API skeleton is ready; real model artifacts are not integrated yet."),
    )


@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Return a Phase 1 mock prediction",
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
        model_version="mock-phase-1",
        is_mock=True,
    )
