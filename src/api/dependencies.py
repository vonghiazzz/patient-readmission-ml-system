"""FastAPI dependencies for the backend service.

Phase 1 intentionally does not load a trained model.

Real model loading will be implemented after Member B exports:
- model artifact
- preprocessing artifact
- model metadata
"""

from src.config.settings import Settings, get_settings


def get_settings_dependency() -> Settings:
    """Provide application settings through dependency injection."""

    return get_settings()
