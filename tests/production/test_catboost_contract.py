import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from src.api.catboost_dependencies import (
    CATBOOST_FEATURES,
    get_catboost_artifacts_dependency,
    load_catboost_artifacts,
)
from src.api.main import app
from src.api.schemas import CatBoostPredictionRequest
from src.config.settings import get_settings


def sample_payload() -> dict[str, Any]:
    value = json.loads(Path("docs/api/sample_catboost_request.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_catboost_artifact_and_request_schema_match_exactly() -> None:
    artifacts = load_catboost_artifacts(get_settings().catboost_model_path)
    assert artifacts.feature_names == CATBOOST_FEATURES
    assert list(CatBoostPredictionRequest.model_fields) == list(CATBOOST_FEATURES)
    assert list(sample_payload()) == list(CATBOOST_FEATURES)
    assert artifacts.decision_threshold == 0.5
    assert artifacts.sha256 == ("4d5c1217e3f07d976d16b98d946639a742c26e4137066f7e6853b9c67cd05162")


def test_catboost_real_artifact_returns_prediction_zero() -> None:
    with TestClient(app) as client:
        response = client.post("/predict/catboost", json=sample_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "CatBoostClassifier"
    assert body["decision_threshold"] == 0.5
    assert body["prediction"] == int(body["risk_score"] >= 0.5)
    assert body["prediction"] == 0


def test_catboost_prediction_one_uses_artifact_threshold() -> None:
    class PositiveModel:
        def predict_proba(self, _frame: Any) -> np.ndarray:
            return np.array([[0.25, 0.75]])

    artifacts = load_catboost_artifacts(get_settings().catboost_model_path)
    positive_artifacts = replace(artifacts, model=PositiveModel())
    app.dependency_overrides[get_catboost_artifacts_dependency] = lambda: positive_artifacts
    try:
        with TestClient(app) as client:
            response = client.post("/predict/catboost", json=sample_payload())
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["risk_score"] == 0.75
    assert response.json()["prediction"] == 1
    assert response.json()["status"] == "high_risk"


def test_catboost_rejects_extra_input_without_echoing_value() -> None:
    payload = {**sample_payload(), "patient_nbr": "private-value-987654"}
    with TestClient(app) as client:
        response = client.post("/predict/catboost", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "private-value-987654" not in response.text


def test_catboost_unknown_category_is_supported() -> None:
    payload = sample_payload()
    payload["race"] = "PreviouslyUnseenSyntheticCategory"
    with TestClient(app) as client:
        response = client.post("/predict/catboost", json=payload)
    assert response.status_code == 200


def test_catboost_unavailable_returns_safe_503() -> None:
    with TestClient(app) as client:
        loaded = app.state.catboost_artifacts
        app.state.catboost_artifacts = None
        try:
            response = client.post("/predict/catboost", json=sample_payload())
        finally:
            app.state.catboost_artifacts = loaded
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
    assert "/Users/" not in response.text


def test_openapi_exposes_both_prediction_models() -> None:
    paths = app.openapi()["paths"]
    assert "/predict" in paths
    assert "/predict/catboost" in paths
    operation = paths["/predict/catboost"]["post"]
    assert operation["operationId"] == "predictReadmissionRiskWithCatBoost"
    assert {"200", "422", "500", "503"}.issubset(operation["responses"])
