from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings shared by the API service."""

    app_name: str = "Patient Readmission API"
    app_version: str = "0.1.0"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Reserved for the real artifacts delivered in Phase 3.
    model_path: Path = Path("models/model.joblib")
    preprocessor_path: Path = Path("models/preprocessor.joblib")
    metadata_path: Path = Path("models/metadata.json")

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached Settings instance per process."""

    return Settings()
