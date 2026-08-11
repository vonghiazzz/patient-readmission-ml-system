from dataclasses import replace
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_production_artifacts_dependency,
    load_production_artifacts,
)
from src.api.main import app
from src.config.settings import get_settings


def artifact_payload() -> dict[str, Any]:
    artifacts = load_production_artifacts(get_settings().production_artifact_dir)
    payload: dict[str, Any] = {}
    for name, transformer, columns in artifacts.preprocessor.transformers_:
        if name == "remainder" or isinstance(columns, str):
            continue
        if name == "numeric":
            for column in columns:
                if column in artifacts.feature_manifest["request_features"]:
                    payload[column] = 1
            continue
        encoder = transformer.named_steps["encoder"]
        for column, categories in zip(columns, encoder.categories_, strict=True):
            if column not in artifacts.feature_manifest["request_features"]:
                continue
            value = next(value for value in categories.tolist() if str(value) != "nan")
            payload[column] = value.item() if hasattr(value, "item") else value
    return {name: payload[name] for name in artifacts.feature_manifest["request_features"]}


def test_health_returns_process_status() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "Patient Readmission API"
    assert response.json()["version"] == get_settings().app_version


def test_ready_requires_valid_loaded_artifacts() -> None:
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["contract_validated"] is True
    assert response.json()["model_version"] == "1.0.0"


def test_predict_happy_path_uses_frozen_champion() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=artifact_payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "model_version",
        "risk_score",
        "decision_threshold",
        "prediction",
        "status",
    }
    assert body["model_version"] == "1.0.0"
    assert body["decision_threshold"] == 0.17
    assert body["prediction"] == int(body["risk_score"] >= body["decision_threshold"])


def test_predict_rejects_malformed_or_extra_input_without_echoing_value() -> None:
    payload = artifact_payload()
    payload.pop("number_inpatient")
    payload["patient_nbr"] = "private-value-987654"
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "private-value-987654" not in response.text


def test_threshold_boundary_comes_from_bundle_metadata() -> None:
    class BoundaryModel:
        def predict_proba(self, _transformed: Any) -> np.ndarray:
            return np.array([[0.27, 0.73]])

    artifacts = load_production_artifacts(get_settings().production_artifact_dir)
    metadata = {**artifacts.metadata, "decision_threshold": 0.73}
    boundary_artifacts = replace(artifacts, model=BoundaryModel(), metadata=metadata)
    app.dependency_overrides[get_production_artifacts_dependency] = lambda: boundary_artifacts
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=artifact_payload())
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["risk_score"] == 0.73
    assert response.json()["decision_threshold"] == 0.73
    assert response.json()["prediction"] == 1


def test_openapi_exposes_production_endpoint_and_error_contracts() -> None:
    operation = app.openapi()["paths"]["/predict"]["post"]
    responses = operation["responses"]
    assert {"200", "422", "500", "503"}.issubset(responses)
    assert operation["operationId"] == "predictReadmissionRisk"


def test_compatibility_prediction_alias_uses_same_contract() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=artifact_payload())
    assert response.status_code == 200
    assert response.json()["model_version"] == "1.0.0"
    assert response.json()["decision_threshold"] == 0.17


def test_metrics_exposes_privacy_safe_request_and_prediction_metrics() -> None:
    with TestClient(app) as client:
        assert client.post("/predict", json=artifact_payload()).status_code == 200
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "readmission_http_requests_total" in response.text
    assert "readmission_http_request_duration_seconds" in response.text
    assert "readmission_predictions_total" in response.text
    assert "readmission_prediction_risk_score_bucket" in response.text
    assert 'model_version="1.0.0"' in response.text
    assert "patient_nbr" not in response.text
    assert "encounter_id" not in response.text


def test_unavailable_artifacts_make_ready_and_predict_return_503() -> None:
    with TestClient(app) as client:
        loaded = app.state.production_artifacts
        app.state.production_artifacts = None
        try:
            ready_response = client.get("/ready")
            predict_response = client.post("/predict", json=artifact_payload())
        finally:
            app.state.production_artifacts = loaded
    assert ready_response.status_code == 503
    assert ready_response.json()["status"] == "not_ready"
    assert predict_response.status_code == 503
    assert predict_response.json()["error"]["code"] == "model_unavailable"
    assert "/Users/" not in predict_response.text


def test_unexpected_inference_error_is_privacy_safe() -> None:
    class FailingModel:
        def predict_proba(self, _transformed: Any) -> np.ndarray:
            raise RuntimeError("private-value-987654 /internal/model/path")

    artifacts = load_production_artifacts(get_settings().production_artifact_dir)
    failing_artifacts = replace(artifacts, model=FailingModel())
    app.dependency_overrides[get_production_artifacts_dependency] = lambda: failing_artifacts
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/predict", json=artifact_payload())
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_server_error"
    assert "private-value-987654" not in response.text
    assert "/internal/model/path" not in response.text
