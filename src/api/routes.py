from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from src.api.dependencies import (
    ProductionArtifacts,
    get_production_artifacts_dependency,
    get_settings_dependency,
)
from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)
from src.config.settings import Settings
from src.monitoring.metrics import metrics_response, record_prediction, set_model_readiness

router = APIRouter()
SettingsDependency = Annotated[Settings, Depends(get_settings_dependency)]
ArtifactsDependency = Annotated[
    ProductionArtifacts,
    Depends(get_production_artifacts_dependency),
]


@router.get(
    "/health",
    operation_id="getServiceHealth",
    response_model=HealthResponse,
    tags=["Operations"],
)
def health(settings: SettingsDependency) -> HealthResponse:
    return HealthResponse(status="healthy", service=settings.app_name, version=settings.app_version)


@router.get(
    "/ready",
    operation_id="getModelReadiness",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
    tags=["Operations"],
)
def readiness(request: Request, response: Response) -> ReadinessResponse:
    artifacts = getattr(request.app.state, "production_artifacts", None)
    if not isinstance(artifacts, ProductionArtifacts):
        set_model_readiness(False)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            model_loaded=False,
            contract_validated=False,
            message="Huy production artifacts are unavailable or invalid.",
        )

    feature_set = str(artifacts.feature_manifest["feature_set"])
    set_model_readiness(True, artifacts.model_version, feature_set)
    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        contract_validated=True,
        model_version=artifacts.model_version,
        feature_set=feature_set,
        message="Huy final CatBoost model is ready for inference.",
    )


@router.post("/api/v1/predict", response_model=PredictionResponse, include_in_schema=False)
@router.post(
    "/predict",
    operation_id="predictReadmissionRisk",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Request schema validation failed."},
        500: {"model": ErrorResponse, "description": "Unexpected server error."},
        503: {"model": ErrorResponse, "description": "Huy model artifacts are unavailable."},
    },
    tags=["Prediction"],
    summary="Predict 30-day readmission risk with Huy's final CatBoost model",
    description=(
        "Accepts 40 raw encounter fields, reproduces Huy's 52-feature preprocessing, "
        "and returns the raw CatBoost probability. prediction is 1 when risk_score is "
        "greater than or equal to the notebook threshold 0.8564852152742759."
    ),
)
def predict(
    request: PredictionRequest,
    artifacts: ArtifactsDependency,
) -> PredictionResponse:
    result = artifacts.predict(request.model_dump())
    record_prediction(result.risk_score, result.prediction, model="huy_catboost")
    return PredictionResponse(
        model_version=artifacts.model_version,
        risk_score=result.risk_score,
        decision_threshold=artifacts.decision_threshold,
        prediction=result.prediction,
        status="high_risk" if result.prediction else "not_high_risk",
    )


@router.get(
    "/metrics",
    operation_id="getPrometheusMetrics",
    include_in_schema=False,
    tags=["Operations"],
)
def metrics() -> Response:
    return metrics_response()
