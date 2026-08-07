from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.routes import router
from src.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "API for predicting the probability of hospital readmission "
        "within 30 days. Phase 2 currently exposes an explicitly "
        "marked mock prediction response."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

register_exception_handlers(app)
app.include_router(router)
