from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import ArtifactContractError, load_production_artifacts
from src.api.exception_handlers import register_exception_handlers
from src.api.routes import router
from src.config.settings import get_settings
from src.monitoring.metrics import configure_model_info, http_metrics_middleware

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.production_artifacts = load_production_artifacts(settings.production_artifact_dir)
        app.state.artifact_error = None
        artifacts = app.state.production_artifacts
        configure_model_info(
            ready=True,
            model_version=artifacts.model_version,
            feature_set=str(artifacts.feature_manifest["feature_set"]),
        )
    except ArtifactContractError as exception:
        app.state.production_artifacts = None
        app.state.artifact_error = type(exception).__name__
        configure_model_info(ready=False)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "API for relative 30-day hospital readmission risk using Huy's final CatBoost "
        "model. It is a decision-support demonstration, not an autonomous clinical system."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.middleware("http")(http_metrics_middleware)
app.include_router(router)
