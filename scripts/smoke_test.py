"""Privacy-safe smoke test for a running Patient Readmission API."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, str]:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    outgoing = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(outgoing, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def assert_json(status: int, body: str, expected_status: int) -> dict[str, Any]:
    if status != expected_status:
        raise RuntimeError(f"Expected HTTP {expected_status}, received {status}: {body}")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("Expected a JSON object response")
    return parsed


def run(base_url: str, sample_path: Path, catboost_sample_path: Path) -> None:
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    catboost_payload = json.loads(catboost_sample_path.read_text(encoding="utf-8"))

    status, body = request(base_url, "/health")
    health = assert_json(status, body, 200)
    assert health["status"] == "healthy"

    status, body = request(base_url, "/ready")
    ready = assert_json(status, body, 200)
    assert ready["model_version"] == "1.0.0"
    assert ready["contract_validated"] is True

    for endpoint in ("/predict", "/api/v1/predict"):
        status, body = request(base_url, endpoint, payload)
        prediction = assert_json(status, body, 200)
        assert prediction["model_version"] == "1.0.0"
        assert prediction["decision_threshold"] == 0.17
        assert prediction["prediction"] == int(prediction["risk_score"] >= 0.17)

    status, body = request(base_url, "/predict/catboost", catboost_payload)
    catboost_prediction = assert_json(status, body, 200)
    assert catboost_prediction["model_type"] == "CatBoostClassifier"
    assert catboost_prediction["decision_threshold"] == 0.5
    assert catboost_prediction["prediction"] == int(catboost_prediction["risk_score"] >= 0.5)

    status, body = request(base_url, "/predict", {"patient_nbr": "not-accepted"})
    error = assert_json(status, body, 422)
    assert error["error"]["code"] == "validation_error"
    assert "not-accepted" not in body

    status, body = request(base_url, "/metrics")
    if status != 200 or "readmission_model_ready 1.0" not in body:
        raise RuntimeError("Prometheus metrics or readiness gauge is unavailable")
    if "readmission_predictions_total" not in body:
        raise RuntimeError("Prediction metrics were not recorded")

    print("API smoke passed: health, readiness, XGBoost, CatBoost, alias, invalid request, metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("docs/api/sample_request.json"),
    )
    parser.add_argument(
        "--catboost-sample",
        type=Path,
        default=Path("docs/api/sample_catboost_request.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.base_url, arguments.sample, arguments.catboost_sample)
