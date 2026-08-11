import json
import os
import subprocess
import sys
from pathlib import Path


def test_missing_runtime_artifact_path_keeps_health_observable(tmp_path: Path) -> None:
    payload = json.loads(Path("docs/api/sample_request.json").read_text(encoding="utf-8"))
    code = f"""
import json
from fastapi.testclient import TestClient
from src.api.main import app
payload = json.loads({json.dumps(json.dumps(payload))})
with TestClient(app) as client:
    print(json.dumps({{
        'health': client.get('/health').status_code,
        'ready': client.get('/ready').status_code,
        'predict': client.post('/predict', json=payload).status_code,
    }}))
"""
    environment = {
        **os.environ,
        "PRODUCTION_ARTIFACT_DIR": str(tmp_path / "unavailable"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    statuses = json.loads(result.stdout.strip().splitlines()[-1])
    assert statuses == {"health": 200, "ready": 503, "predict": 503}
