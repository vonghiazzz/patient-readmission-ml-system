from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Patient Readmission API"
    app_version: str = "2.0.0"
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    production_artifact_dir: Path = Path("models/production_huy")
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
