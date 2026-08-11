"""Validated loading and inference for the separately supplied CatBoost model."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Request

from src.api.dependencies import PROJECT_ROOT, ArtifactContractError, PredictionResult

CATBOOST_FEATURES = (
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "numchange",
    "level1_diag1",
    "nummed",
    "number_inpatient_log1p",
    "number_outpatient_log1p",
    "number_emergency_log1p",
    "service_utilization_log1p",
    "time_in_hospital|num_lab_procedures",
    "num_medications|num_lab_procedures",
    "num_medications|number_diagnoses",
    "age|number_diagnoses",
    "change|num_medications",
    "number_diagnoses|time_in_hospital",
    "num_medications|time_in_hospital_log",
    "num_medications|num_procedures_log1p",
    "num_medications|numchange_log1p",
)

CATBOOST_CATEGORICAL_FEATURES = (
    "race",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "level1_diag1",
)


@dataclass(frozen=True)
class CatBoostArtifacts:
    """Validated runtime view of the unversioned experimental CatBoost artifact."""

    model: Any
    model_path: Path
    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]
    decision_threshold: float
    sha256: str

    def predict(self, payload: Mapping[str, Any]) -> PredictionResult:
        submitted = set(payload)
        expected = set(self.feature_names)
        if submitted != expected:
            raise ArtifactContractError("CatBoost prediction payload contract mismatch")

        frame = pd.DataFrame(
            [{feature: payload[feature] for feature in self.feature_names}],
            columns=self.feature_names,
        )
        probabilities = np.asarray(self.model.predict_proba(frame), dtype=float)
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
            raise ArtifactContractError("CatBoost model returned invalid class probabilities")

        risk_score = float(probabilities[0, 1])
        return PredictionResult(
            risk_score=risk_score,
            prediction=int(risk_score >= self.decision_threshold),
        )


def _resolve_model_path(model_path: Path) -> Path:
    return model_path if model_path.is_absolute() else PROJECT_ROOT / model_path


@lru_cache(maxsize=4)
def load_catboost_artifacts(model_path: Path) -> CatBoostArtifacts:
    """Load the CatBoost pickle once and verify its embedded feature contract."""

    resolved = _resolve_model_path(model_path)
    try:
        artifact_bytes = resolved.read_bytes()
        model = joblib.load(resolved)
    except Exception as exception:
        raise ArtifactContractError("Cannot load CatBoost model artifact") from exception

    feature_names = tuple(str(name) for name in getattr(model, "feature_names_", ()))
    if feature_names != CATBOOST_FEATURES:
        raise ArtifactContractError("CatBoost feature names or order do not match the API contract")
    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactContractError("CatBoost artifact does not expose predict_proba")

    try:
        categorical_indices = tuple(int(index) for index in model.get_cat_feature_indices())
        categorical_features = tuple(feature_names[index] for index in categorical_indices)
        threshold = float(model.get_probability_threshold())
    except Exception as exception:
        raise ArtifactContractError("Cannot inspect CatBoost model contract") from exception
    if categorical_features != CATBOOST_CATEGORICAL_FEATURES:
        raise ArtifactContractError("CatBoost categorical features do not match the API contract")
    if not 0 <= threshold <= 1:
        raise ArtifactContractError("CatBoost probability threshold is invalid")

    return CatBoostArtifacts(
        model=model,
        model_path=resolved,
        feature_names=feature_names,
        categorical_features=categorical_features,
        decision_threshold=threshold,
        sha256=hashlib.sha256(artifact_bytes).hexdigest(),
    )


def get_catboost_artifacts_dependency(request: Request) -> CatBoostArtifacts:
    artifacts = getattr(request.app.state, "catboost_artifacts", None)
    if not isinstance(artifacts, CatBoostArtifacts):
        raise ArtifactContractError("CatBoost model is unavailable")
    return artifacts
