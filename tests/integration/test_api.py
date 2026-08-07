from fastapi.testclient import TestClient

from src.api.dependencies import get_settings_dependency
from src.api.exception_handlers import ModelUnavailableError
from src.api.main import app

client = TestClient(app)


def valid_prediction_payload() -> dict[str, object]:
    return {
        "race": "Caucasian",
        "age": "[60-70)",
        "gender": "Female",
        "time_in_hospital": 4,
        "num_lab_procedures": 42,
        "num_procedures": 1,
        "num_medications": 12,
        "number_inpatient": 1,
        "number_emergency": 0,
        "number_outpatient": 2,
        "number_diagnoses": 7,
    }


def test_health_returns_200() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "Patient Readmission API"
    assert body["version"] == "0.1.0"


def test_readiness_identifies_mock_mode() -> None:
    response = client.get("/ready")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ready"
    assert body["mode"] == "mock"
    assert body["model_loaded"] is False


def test_predict_returns_explicit_mock_response() -> None:
    response = client.post(
        "/api/v1/predict",
        json=valid_prediction_payload(),
    )

    assert response.status_code == 200

    body = response.json()
    assert body["is_mock"] is True
    assert body["model_version"] == "mock-phase-2"
    assert body["risk_band"] == "mock"


def test_predict_rejects_negative_counts() -> None:
    payload = valid_prediction_payload()
    payload["number_inpatient"] = -1

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"][0]["location"] == ["body", "number_inpatient"]


def test_predict_rejects_missing_required_field() -> None:
    payload = valid_prediction_payload()
    del payload["number_outpatient"]

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_validation_error_does_not_echo_submitted_value() -> None:
    payload = valid_prediction_payload()
    payload["patient_nbr"] = "private-value-987654"

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422
    assert "private-value-987654" not in response.text


def test_unhandled_failure_returns_safe_500() -> None:
    def fail_settings() -> None:
        raise RuntimeError("private-stack-marker")

    app.dependency_overrides[get_settings_dependency] = fail_settings
    try:
        with TestClient(app, raise_server_exceptions=False) as safe_client:
            response = safe_client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_server_error"
    assert "private-stack-marker" not in response.text


def test_model_unavailable_failure_returns_safe_503() -> None:
    def unavailable_settings() -> None:
        raise ModelUnavailableError("private-model-path")

    app.dependency_overrides[get_settings_dependency] = unavailable_settings
    try:
        with TestClient(app, raise_server_exceptions=False) as safe_client:
            response = safe_client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
    assert "private-model-path" not in response.text


def test_openapi_documents_prediction_error_responses() -> None:
    responses = app.openapi()["paths"]["/api/v1/predict"]["post"]["responses"]

    assert {"200", "422", "500", "503"}.issubset(responses)
