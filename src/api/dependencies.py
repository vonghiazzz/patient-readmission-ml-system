"""Loading, contract validation, and inference for Huy's CatBoost champion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Request

from src.config.settings import Settings, get_settings
from src.features.build_features import (
    CATEGORICAL_MODEL_FEATURES,
    MODEL_INPUT_FEATURES,
    REQUEST_FEATURES,
    build_huy_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODEL_VERSION = "huy-catboost-1.0.0"


class ArtifactContractError(RuntimeError):
    """Raised when Huy production artifacts are missing or inconsistent."""


@dataclass(frozen=True)
class PredictionResult:
    risk_score: float
    prediction: int


@dataclass(frozen=True)
class ProductionArtifacts:
    model: Any
    feature_manifest: Mapping[str, Any]
    preprocessing_state: Mapping[str, Any]
    metadata: Mapping[str, Any]
    artifact_dir: Path
    model_sha256: str

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def decision_threshold(self) -> float:
        return float(self.metadata["decision_threshold"])

    def prepare_model_input(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        source = pd.DataFrame(
            [{feature: payload[feature] for feature in REQUEST_FEATURES}],
            columns=REQUEST_FEATURES,
        )
        model_input = build_huy_features(source, self.preprocessing_state)
        if tuple(model_input.columns) != MODEL_INPUT_FEATURES:
            raise ArtifactContractError("Huy model input order does not match its manifest")
        numeric = model_input.drop(columns=list(CATEGORICAL_MODEL_FEATURES)).to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ArtifactContractError("Huy preprocessing returned non-finite numeric features")
        return model_input

    def predict(self, payload: Mapping[str, Any]) -> PredictionResult:
        model_input = self.prepare_model_input(payload)
        probabilities = np.asarray(self.model.predict_proba(model_input), dtype=float)
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
            raise ArtifactContractError("Huy CatBoost model returned invalid probabilities")
        risk_score = float(probabilities[0, 1])
        return PredictionResult(
            risk_score=risk_score,
            prediction=classify_probability(risk_score, self.decision_threshold),
        )


def classify_probability(probability: float, threshold: float) -> int:
    return int(probability >= threshold)


def _resolve_artifact_dir(artifact_dir: Path) -> Path:
    return artifact_dir if artifact_dir.is_absolute() else PROJECT_ROOT / artifact_dir


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ArtifactContractError(f"Cannot read production contract: {path.name}") from exception
    if not isinstance(value, dict):
        raise ArtifactContractError(f"Production contract must be an object: {path.name}")
    return value


def validate_artifact_contract(
    model: Any,
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    if metadata.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ArtifactContractError("Unexpected Huy production model version")
    if metadata.get("feature_set") != manifest.get("feature_set"):
        raise ArtifactContractError("Metadata and Huy feature manifest disagree")
    if tuple(manifest.get("request_features", ())) != REQUEST_FEATURES:
        raise ArtifactContractError("Huy request features or order are invalid")
    if tuple(manifest.get("model_input_features", ())) != MODEL_INPUT_FEATURES:
        raise ArtifactContractError("Huy model features or order are invalid")
    if tuple(manifest.get("categorical_model_features", ())) != CATEGORICAL_MODEL_FEATURES:
        raise ArtifactContractError("Huy categorical features or order are invalid")

    threshold = metadata.get("decision_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ArtifactContractError("Decision threshold must be numeric")
    if not 0 <= float(threshold) <= 1:
        raise ArtifactContractError("Decision threshold must be between zero and one")

    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactContractError("Huy CatBoost artifact does not expose predict_proba")
    if tuple(str(name) for name in getattr(model, "feature_names_", ())) != MODEL_INPUT_FEATURES:
        raise ArtifactContractError("CatBoost embedded feature names do not match Huy contract")
    categorical_indices = tuple(int(index) for index in model.get_cat_feature_indices())
    categorical_names = tuple(MODEL_INPUT_FEATURES[index] for index in categorical_indices)
    if categorical_names != CATEGORICAL_MODEL_FEATURES:
        raise ArtifactContractError("CatBoost categorical indices do not match Huy contract")

    for section in ("standard_1", "minmax_1", "standard_2", "age_minmax"):
        if section not in state:
            raise ArtifactContractError(f"Huy preprocessing state is missing {section}")


@lru_cache(maxsize=4)
def load_production_artifacts(artifact_dir: Path) -> ProductionArtifacts:
    resolved = _resolve_artifact_dir(artifact_dir)
    manifest = _read_json(resolved / "feature_manifest.json")
    state = _read_json(resolved / "preprocessing_state.json")
    metadata = _read_json(resolved / "metadata.json")
    model_path = resolved / "model.pkl"
    try:
        model_bytes = model_path.read_bytes()
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        if metadata.get("model_sha256") != model_sha256:
            raise ArtifactContractError("Huy CatBoost model checksum does not match metadata")
        model = joblib.load(model_path)
    except ArtifactContractError:
        raise
    except Exception as exception:
        raise ArtifactContractError("Cannot load Huy CatBoost model") from exception

    validate_artifact_contract(model, manifest, state, metadata)
    return ProductionArtifacts(
        model=model,
        feature_manifest=manifest,
        preprocessing_state=state,
        metadata=metadata,
        artifact_dir=resolved,
        model_sha256=model_sha256,
    )


def get_settings_dependency() -> Settings:
    return get_settings()


def get_production_artifacts_dependency(request: Request) -> ProductionArtifacts:
    artifacts = getattr(request.app.state, "production_artifacts", None)
    if not isinstance(artifacts, ProductionArtifacts):
        raise ArtifactContractError("Huy production artifacts are not ready")
    return artifacts
