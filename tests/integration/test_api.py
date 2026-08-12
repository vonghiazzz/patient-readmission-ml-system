import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from src.api.dependencies import get_production_artifacts_dependency
from src.api.main import app


def payload(filename: str = "sample_request.json") -> dict[str, Any]:
    return json.loads(Path("docs/api", filename).read_text(encoding="utf-8"))


def test_health_and_huy_readiness() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
    assert health.status_code == 200
    assert health.json()["version"] == "2.0.0"
    assert ready.status_code == 200
    assert ready.json()["model_version"] == "huy-catboost-1.0.0"
    assert ready.json()["feature_set"] == "HUY_FINAL_52"


def test_predict_returns_huy_reference_zero_and_one() -> None:
    with TestClient(app) as client:
        low = client.post("/predict", json=payload())
        high = client.post("/predict", json=payload("sample_high_risk_request.json"))
    assert low.status_code == 200
    assert high.status_code == 200
    assert low.json()["prediction"] == 0
    assert high.json()["prediction"] == 1
    assert low.json()["decision_threshold"] == 0.8564852152742759
    assert low.json()["model_version"] == "huy-catboost-1.0.0"


def test_compatibility_alias_serves_the_same_huy_model() -> None:
    with TestClient(app) as client:
        direct = client.post("/predict", json=payload()).json()
        alias = client.post("/api/v1/predict", json=payload()).json()
    assert alias == direct


def test_invalid_or_private_extra_input_is_rejected_without_echo() -> None:
    invalid = {**payload(), "patient_nbr": "private-value-987654"}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "private-value-987654" not in response.text


def test_openapi_exposes_one_prediction_story() -> None:
    paths = app.openapi()["paths"]
    assert "/predict" in paths
    assert "/predict/catboost" not in paths
    operation = paths["/predict"]["post"]
    assert operation["operationId"] == "predictReadmissionRisk"
    assert "Huy" in operation["summary"]


def test_prediction_uses_metadata_threshold() -> None:
    class BoundaryModel:
        def predict_proba(self, _frame: Any) -> np.ndarray:
            return np.array([[0.1435147847257241, 0.8564852152742759]])

    with TestClient(app) as client:
        changed = replace(client.app.state.production_artifacts, model=BoundaryModel())
        app.dependency_overrides[get_production_artifacts_dependency] = lambda: changed
        try:
            response = client.post("/predict", json=payload())
        finally:
            app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["prediction"] == 1


def test_metrics_do_not_expose_patient_fields() -> None:
    with TestClient(app) as client:
        client.post("/predict", json=payload())
        response = client.get("/metrics")
    assert response.status_code == 200
    assert 'model="huy_catboost"' in response.text
    assert "patient_nbr" not in response.text
    assert "encounter_id" not in response.text
