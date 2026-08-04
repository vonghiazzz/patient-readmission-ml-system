from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


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
        json={
            "age": "[60-70)",
            "gender": "Female",
            "number_inpatient": 1,
            "number_emergency": 0,
            "number_outpatient": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["is_mock"] is True
    assert body["model_version"] == "mock-phase-1"
    assert body["risk_band"] == "mock"


def test_predict_rejects_negative_counts() -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "age": "[60-70)",
            "gender": "Female",
            "number_inpatient": -1,
            "number_emergency": 0,
            "number_outpatient": 2,
        },
    )

    assert response.status_code == 422


def test_predict_rejects_missing_required_field() -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "age": "[60-70)",
            "gender": "Female",
            "number_inpatient": 1,
            "number_emergency": 0,
        },
    )

    assert response.status_code == 422
