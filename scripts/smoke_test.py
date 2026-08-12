"""Privacy-safe smoke test for the Huy CatBoost prediction service."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL_VERSION = "huy-catboost-1.0.0"
DECISION_THRESHOLD = 0.8564852152742759


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


def run(base_url: str, low_sample: Path, high_sample: Path) -> None:
    low_payload = json.loads(low_sample.read_text(encoding="utf-8"))
    high_payload = json.loads(high_sample.read_text(encoding="utf-8"))

    status, body = request(base_url, "/health")
    assert assert_json(status, body, 200)["status"] == "healthy"

    status, body = request(base_url, "/ready")
    ready = assert_json(status, body, 200)
    assert ready["model_version"] == MODEL_VERSION
    assert ready["feature_set"] == "HUY_FINAL_52"

    for endpoint in ("/predict", "/api/v1/predict"):
        status, body = request(base_url, endpoint, low_payload)
        prediction = assert_json(status, body, 200)
        assert prediction["model_version"] == MODEL_VERSION
        assert prediction["decision_threshold"] == DECISION_THRESHOLD
        assert prediction["prediction"] == 0

    status, body = request(base_url, "/predict", high_payload)
    assert assert_json(status, body, 200)["prediction"] == 1

    status, body = request(base_url, "/predict", {"patient_nbr": "not-accepted"})
    error = assert_json(status, body, 422)
    assert error["error"]["code"] == "validation_error"
    assert "not-accepted" not in body

    status, body = request(base_url, "/metrics")
    if status != 200 or "readmission_model_ready 1.0" not in body:
        raise RuntimeError("Prometheus metrics or readiness gauge is unavailable")
    if 'model="huy_catboost"' not in body:
        raise RuntimeError("Huy prediction metrics were not recorded")

    print("API smoke passed: Huy CatBoost readiness, prediction 0/1, validation, metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--low-sample", type=Path, default=Path("docs/api/sample_request.json"))
    parser.add_argument(
        "--high-sample",
        type=Path,
        default=Path("docs/api/sample_high_risk_request.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.base_url, arguments.low_sample, arguments.high_sample)
