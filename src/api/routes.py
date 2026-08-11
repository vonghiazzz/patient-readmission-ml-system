from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from src.api.catboost_dependencies import (
    CatBoostArtifacts,
    get_catboost_artifacts_dependency,
)
from src.api.dependencies import (
    ProductionArtifacts,
    get_production_artifacts_dependency,
    get_settings_dependency,
)
from src.api.schemas import (
    CatBoostPredictionRequest,
    CatBoostPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)
from src.config.settings import Settings
from src.monitoring.metrics import (
    metrics_response,
    record_prediction,
    set_model_readiness,
)

router = APIRouter()

SettingsDependency = Annotated[Settings, Depends(get_settings_dependency)]
ArtifactsDependency = Annotated[
    ProductionArtifacts,
    Depends(get_production_artifacts_dependency),
]
CatBoostArtifactsDependency = Annotated[
    CatBoostArtifacts,
    Depends(get_catboost_artifacts_dependency),
]


@router.get(
    "/health",
    operation_id="getServiceHealth",
    response_model=HealthResponse,
    tags=["Operations"],
)
def health(settings: SettingsDependency) -> HealthResponse:
    """Confirm that the API process is alive; this does not imply model readiness."""

    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get(
    "/ready",
    operation_id="getModelReadiness",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
    tags=["Operations"],
)
def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Report ready only after all four artifacts and their contract validate."""

    artifacts = getattr(request.app.state, "production_artifacts", None)
    if not isinstance(artifacts, ProductionArtifacts):
        set_model_readiness(False)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            model_loaded=False,
            contract_validated=False,
            message="Production artifacts are unavailable or invalid.",
        )

    set_model_readiness(
        True,
        artifacts.model_version,
        str(artifacts.feature_manifest["feature_set"]),
    )
    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        contract_validated=True,
        model_version=artifacts.model_version,
        feature_set=str(artifacts.feature_manifest["feature_set"]),
        message="Frozen production model is ready for inference.",
    )


@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    include_in_schema=False,
)
@router.post(
    "/predict",
    operation_id="predictReadmissionRisk",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Request schema validation failed."},
        500: {"model": ErrorResponse, "description": "Unexpected server error."},
        503: {"model": ErrorResponse, "description": "Production artifacts are unavailable."},
    },
    tags=["Prediction"],
    summary="Predict 30-day readmission risk with frozen champion 1.0.0",
    description=(
        "Returns the frozen XGBoost raw predict_proba score for 30-day readmission. "
        "prediction is 1 exactly when risk_score is greater than or equal to the "
        "decision_threshold loaded from metadata. The score is not post-hoc calibrated."
    ),
)
def predict(
    request: PredictionRequest,
    artifacts: ArtifactsDependency,
) -> PredictionResponse:
    """Run manifest-ordered preprocessing and raw XGBoost probability inference."""

    result = artifacts.predict(request.model_dump())
    record_prediction(result.risk_score, result.prediction, model="xgboost_v1")
    return PredictionResponse(
        model_version=artifacts.model_version,
        risk_score=result.risk_score,
        decision_threshold=artifacts.decision_threshold,
        prediction=result.prediction,
        status="high_risk" if result.prediction else "not_high_risk",
    )


@router.post(
    "/predict/catboost",
    operation_id="predictReadmissionRiskWithCatBoost",
    response_model=CatBoostPredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Request schema validation failed."},
        500: {"model": ErrorResponse, "description": "Unexpected server error."},
        503: {"model": ErrorResponse, "description": "CatBoost artifact is unavailable."},
    },
    tags=["Prediction"],
    summary="Predict readmission risk with the experimental CatBoost artifact",
    description=(
        "Accepts exactly the 52 already-engineered features embedded in "
        "cat_tunning_model.pkl. This endpoint is independent of the frozen XGBoost V1 "
        "champion and uses the CatBoost artifact's probability threshold (0.5)."
    ),
)
def predict_catboost(
    request: CatBoostPredictionRequest,
    artifacts: CatBoostArtifactsDependency,
) -> CatBoostPredictionResponse:
    result = artifacts.predict(request.model_dump())
    record_prediction(result.risk_score, result.prediction, model="catboost_experimental")
    return CatBoostPredictionResponse(
        model_type="CatBoostClassifier",
        artifact_sha256=artifacts.sha256,
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
    """Expose low-cardinality, privacy-safe Prometheus metrics."""

    return metrics_response()
