from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.catboost_dependencies import load_catboost_artifacts
from src.api.dependencies import ArtifactContractError, load_production_artifacts
from src.api.exception_handlers import register_exception_handlers
from src.api.routes import router
from src.config.settings import get_settings
from src.monitoring.metrics import configure_model_info, http_metrics_middleware

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load artifacts while keeping /health and /ready observable on failure."""

    try:
        app.state.production_artifacts = load_production_artifacts(settings.production_artifact_dir)
        app.state.artifact_error = None
        configure_model_info(
            ready=True,
            model_version=app.state.production_artifacts.model_version,
            feature_set=str(app.state.production_artifacts.feature_manifest["feature_set"]),
        )
    except ArtifactContractError as exception:
        app.state.production_artifacts = None
        app.state.artifact_error = type(exception).__name__
        configure_model_info(ready=False)

    try:
        app.state.catboost_artifacts = load_catboost_artifacts(settings.catboost_model_path)
        app.state.catboost_error = None
    except ArtifactContractError as exception:
        app.state.catboost_artifacts = None
        app.state.catboost_error = type(exception).__name__
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "API for relative 30-day hospital readmission risk. /predict serves the frozen "
        "XGBoost V1 champion; /predict/catboost serves a separately supplied experimental "
        "CatBoost artifact. It is not an autonomous clinical system."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

register_exception_handlers(app)
app.middleware("http")(http_metrics_middleware)
app.include_router(router)
