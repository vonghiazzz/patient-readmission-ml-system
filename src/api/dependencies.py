"""Central loading, validation, and inference for frozen production artifacts."""

from __future__ import annotations

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
from src.features.build_features import DERIVED_FEATURES, build_v1_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_SCHEMA_ARTIFACT_DIR = PROJECT_ROOT / "models" / "production_v1"
EXPECTED_MODEL_VERSION = "1.0.0"


class ArtifactContractError(RuntimeError):
    """Raised when frozen artifacts are missing, unreadable, or inconsistent."""


@dataclass(frozen=True)
class PredictionResult:
    risk_score: float
    prediction: int


@dataclass(frozen=True)
class ProductionArtifacts:
    """A validated, immutable view of the four production artifacts."""

    model: Any
    preprocessor: Any
    feature_manifest: Mapping[str, Any]
    metadata: Mapping[str, Any]
    artifact_dir: Path
    transformed_feature_count: int

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def decision_threshold(self) -> float:
        return float(self.metadata["decision_threshold"])

    def prepare_model_input(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        """Create the exact manifest-ordered 45-column model input."""

        request_features = list(self.feature_manifest["request_features"])
        submitted = set(payload)
        expected = set(request_features)
        if submitted != expected:
            missing = sorted(expected - submitted)
            extra = sorted(submitted - expected)
            raise ArtifactContractError(
                f"Prediction payload contract mismatch; missing={missing}, extra={extra}"
            )

        source = pd.DataFrame([{name: payload[name] for name in request_features}])
        source = source.where(source.notna(), np.nan)
        model_input = build_v1_features(source)
        model_features = list(self.feature_manifest["model_input_features"])
        model_input = model_input.loc[:, model_features]
        if model_input.columns.tolist() != model_features:
            raise ArtifactContractError("V1 model input order does not match feature manifest")
        return model_input

    def predict(self, payload: Mapping[str, Any]) -> PredictionResult:
        """Run frozen preprocessing and raw ``predict_proba`` inference."""

        model_input = self.prepare_model_input(payload)
        transformed = self.preprocessor.transform(model_input)
        if transformed.shape != (1, self.transformed_feature_count):
            raise ArtifactContractError(
                "Frozen preprocessor returned an unexpected transformed shape"
            )

        probabilities = np.asarray(self.model.predict_proba(transformed), dtype=float)
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
            raise ArtifactContractError("Frozen model returned invalid class probabilities")

        risk_score = float(probabilities[0, 1])
        return PredictionResult(
            risk_score=risk_score,
            prediction=classify_probability(risk_score, self.decision_threshold),
        )


def classify_probability(probability: float, threshold: float) -> int:
    """Apply the inclusive production decision boundary."""

    return int(probability >= threshold)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ArtifactContractError(
            f"Cannot read artifact contract file: {path.name}"
        ) from exception
    if not isinstance(value, dict):
        raise ArtifactContractError(f"Artifact contract must be an object: {path.name}")
    return value


@lru_cache(maxsize=4)
def load_schema_contract(artifact_dir: Path) -> tuple[dict[str, Any], Any]:
    """Load only the manifest and preprocessor needed to construct API types."""

    resolved = _resolve_artifact_dir(artifact_dir)
    manifest = _read_json(resolved / "feature_manifest.json")
    try:
        preprocessor = joblib.load(resolved / "preprocessor.joblib")
    except Exception as exception:
        raise ArtifactContractError("Cannot load frozen preprocessor") from exception
    return manifest, preprocessor


def validate_artifact_contract(
    model: Any,
    preprocessor: Any,
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> int:
    """Validate cross-artifact invariants and return transformed feature count."""

    if metadata.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ArtifactContractError("Unexpected production model version")
    if metadata.get("feature_set") != manifest.get("feature_set"):
        raise ArtifactContractError("Metadata and manifest feature sets disagree")

    sections = {
        "request_features": 42,
        "derived_features": 3,
        "model_input_features": 45,
    }
    for name, expected_count in sections.items():
        values = manifest.get(name)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ArtifactContractError(f"Manifest {name} must contain {expected_count} entries")
        if len(values) != len(set(values)):
            raise ArtifactContractError(f"Manifest {name} contains duplicate entries")

    if tuple(manifest["derived_features"]) != DERIVED_FEATURES:
        raise ArtifactContractError("Manifest does not define the canonical V1 derived features")
    expected_model_inputs = manifest["request_features"] + manifest["derived_features"]
    if manifest["model_input_features"] != expected_model_inputs:
        raise ArtifactContractError(
            "Model inputs are not request features followed by V1 derived features"
        )

    threshold = metadata.get("decision_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ArtifactContractError("Decision threshold must be numeric")
    if not 0 <= float(threshold) <= 1:
        raise ArtifactContractError("Decision threshold must be between zero and one")

    if not callable(getattr(preprocessor, "transform", None)):
        raise ArtifactContractError("Frozen preprocessor does not expose transform")
    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactContractError("Frozen model does not expose predict_proba")

    preprocessor_inputs = list(getattr(preprocessor, "feature_names_in_", []))
    if preprocessor_inputs != manifest["model_input_features"]:
        raise ArtifactContractError("Preprocessor input order disagrees with manifest")
    if int(getattr(preprocessor, "n_features_in_", -1)) != len(preprocessor_inputs):
        raise ArtifactContractError("Preprocessor input feature count is inconsistent")

    try:
        transformed_feature_count = len(preprocessor.get_feature_names_out())
    except Exception as exception:
        raise ArtifactContractError("Cannot inspect preprocessor output features") from exception
    if int(getattr(model, "n_features_in_", -1)) != transformed_feature_count:
        raise ArtifactContractError("Model and preprocessor transformed dimensions disagree")
    return transformed_feature_count


def _resolve_artifact_dir(artifact_dir: Path) -> Path:
    return artifact_dir if artifact_dir.is_absolute() else PROJECT_ROOT / artifact_dir


@lru_cache(maxsize=4)
def load_production_artifacts(artifact_dir: Path) -> ProductionArtifacts:
    """Load and validate the frozen champion exactly once for a given path."""

    resolved = _resolve_artifact_dir(artifact_dir)
    manifest = _read_json(resolved / "feature_manifest.json")
    metadata = _read_json(resolved / "metadata.json")
    try:
        preprocessor = joblib.load(resolved / "preprocessor.joblib")
        model = joblib.load(resolved / "model.joblib")
    except Exception as exception:
        raise ArtifactContractError("Cannot load frozen production model artifacts") from exception

    transformed_feature_count = validate_artifact_contract(
        model=model,
        preprocessor=preprocessor,
        manifest=manifest,
        metadata=metadata,
    )
    return ProductionArtifacts(
        model=model,
        preprocessor=preprocessor,
        feature_manifest=manifest,
        metadata=metadata,
        artifact_dir=resolved,
        transformed_feature_count=transformed_feature_count,
    )


def get_settings_dependency() -> Settings:
    return get_settings()


def get_production_artifacts_dependency(request: Request) -> ProductionArtifacts:
    """Use the artifact bundle loaded once during the FastAPI lifespan."""

    artifacts = getattr(request.app.state, "production_artifacts", None)
    if not isinstance(artifacts, ProductionArtifacts):
        raise ArtifactContractError("Production artifacts are not ready")
    return artifacts
