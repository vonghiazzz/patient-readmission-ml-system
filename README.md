# Patient Readmission ML System

Production-like course project serving the frozen XGBoost V1 champion for relative 30-day hospital
readmission risk. The system is decision support for prioritization and human review—not diagnosis,
treatment advice, or an autonomous clinical system.

## Final frozen contract

| Item | Value |
| --- | --- |
| Champion | XGBoost V1 Optuna |
| Model version | `1.0.0` |
| Best iteration | `493` |
| Decision threshold | `0.17`, loaded from metadata |
| Probability output | Raw `predict_proba`; no post-hoc calibration |
| Input flow | 42 request → 3 derived → 45 model inputs → 223 transformed |

The authoritative files are under `models/production_v1/`: `model.joblib`,
`preprocessor.joblib`, `feature_manifest.json`, and `metadata.json`. Serving never trains, tunes,
resamples, calibrates, or changes the threshold.

## Components

- FastAPI: health, readiness, inference, OpenAPI, and privacy-safe Prometheus metrics.
- Prometheus: API scrape and alert-rule evaluation.
- Grafana: automatically provisioned datasource and dashboard.
- MLflow: persistent tracking for the already-frozen champion; not in the online prediction path.
- GitHub Actions: dependency check, Ruff, tests/coverage, and Docker build.
- Responsible AI: grouped global SHAP, subgroup audit, model card, and limitations.

See [ARCHITECTURE.md](ARCHITECTURE.md), [MODEL_CARD.md](MODEL_CARD.md),
[RESPONSIBLE_AI.md](RESPONSIBLE_AI.md), and [docs/api/API_CONTRACT.md](docs/api/API_CONTRACT.md).

## Local setup

Requirements: Python 3.11.15 and native OpenMP support required by XGBoost on the host OS.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

Copy local configuration only when needed. `.env` is ignored by Git.

```bash
cp .env.example .env
```

## Quality checks

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=60 -q
```

The legacy third-party reference notebook is excluded from Ruff. It is retained only as historical
reference and is not executable production code.

## Run FastAPI locally

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Useful URLs:

- Health: <http://localhost:8000/health>
- Readiness: <http://localhost:8000/ready>
- Swagger: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- OpenAPI: <http://localhost:8000/openapi.json>
- Metrics: <http://localhost:8000/metrics>

Submit the repository's synthetic request:

```bash
curl --fail-with-body -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data @docs/api/sample_request.json
```

The compatibility alias `POST /api/v1/predict` is retained but hidden from Swagger. A successful
response contains `model_version`, `risk_score`, `decision_threshold`, `prediction`, and `status`.
`prediction` is 1 exactly when `risk_score >= 0.17`. The score is not proof of severity, diagnosis,
or guaranteed readmission.

Invalid request example:

```bash
curl -i -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data '{"patient_nbr":"must-not-be-accepted"}'
```

It returns HTTP 422 without echoing the submitted value.

## Docker image

The production image runs as a non-root user and contains the real frozen bundle and reports.

```bash
docker build -t patient-readmission-api:1.0.0 .
docker run --rm -p 8000:8000 --name readmission-api patient-readmission-api:1.0.0
```

In another terminal:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/ready
curl --fail-with-body -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data @docs/api/sample_request.json
curl --fail http://localhost:8000/metrics
```

## Docker Compose stack

```bash
docker compose config
docker compose up --build -d
docker compose ps
docker compose logs --tail=100 api prometheus grafana mlflow mlflow-init
```

Services:

| Service | URL |
| --- | --- |
| API/Swagger | <http://localhost:8000/docs> |
| Prometheus targets | <http://localhost:9090/targets> |
| Prometheus alerts | <http://localhost:9090/alerts> |
| Grafana | <http://localhost:3000> |
| MLflow | <http://localhost:5050> |

Grafana local defaults come from `.env.example` (`admin`/`admin`) and must be replaced outside a
local demonstration. Open the provisioned **Patient Readmission API** dashboard. The Prometheus
datasource uses `http://prometheus:9090`, which is correct inside Compose.

The `mlflow-init` service logs `xgb-v1-production-1.0.0` into experiment
`patient-readmission-production` without retraining. MLflow, Prometheus, and Grafana use named
volumes. Normal restart preserves them:

```bash
docker compose restart
```

Stop without deleting evidence:

```bash
docker compose down
```

Do not use `docker compose down -v` unless deliberately resetting all local monitoring/tracking
state.

## Safe failure demonstration

Do not rename or delete frozen files. Start a separate container with an invalid artifact directory:

```bash
docker run --rm -p 8001:8000 \
  -e PRODUCTION_ARTIFACT_DIR=/app/models/unavailable \
  patient-readmission-api:1.0.0
```

Then `curl -i http://localhost:8001/ready` returns 503. The process remains observable through
`/health`, and `/predict` returns the privacy-safe model-unavailable response.

## MLflow outside Compose

The current local run can be viewed with:

```bash
python -m mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 --port 5000
```

To idempotently log the frozen champion to another tracking server:

```bash
python -m src.evaluation.mlflow_champion \
  --artifact-dir models/production_v1 \
  --tracking-uri http://localhost:5000 \
  --experiment-name patient-readmission-production
```

This command reloads and logs existing artifacts; it does not fit a model.

## Security, privacy, and artifact policy

- `patient_nbr`, `encounter_id`, target fields, diagnoses, weight, and discharge disposition are not
  accepted by the production API.
- Request bodies and demographic/raw feature values are not used as metric labels or logged.
- `.env`, raw/interim patient data, caches, and local MLflow state are ignored by Git.
- The four production artifacts are small enough for normal Git or Git LFS, but the team must choose
  and document one policy. CI requires those files to be available after checkout.

At this audit point `models/production_v1/` is untracked locally. Do not push until the team chooses
normal Git, Git LFS, or an authenticated CI download mechanism.

## XGBoost portability note

The serialized booster records XGBoost `3.3.0`, which requires Python 3.12 or newer. The course
runtime is fixed at Python 3.11.15, so XGBoost is pinned to the newest compatible release, `3.2.0`.
Loading therefore emits XGBoost's cross-version pickle warning. The artifact remains a Python
pickle/joblib object, which is less portable and less safe than XGBoost's native model format. The
frozen artifact is not regenerated; contract tests pin an observed prediction so runtime changes
expose reproducibility regressions.

## Troubleshooting

- `/health` 200 but `/ready` 503: verify the four files and `PRODUCTION_ARTIFACT_DIR`.
- Docker cannot connect: start Docker Desktop/OrbStack before build or Compose commands.
- Prometheus target down: confirm API health and `api:8000/metrics` from the Compose network.
- Grafana has no data: verify Prometheus target `UP`, generate a few predictions, and select the last
  15–30 minutes.
- MLflow init exits 0: this is expected for the one-shot logging service.
- CI artifact preflight fails: settle and implement the production artifact Git/LFS/download policy.

For the exact rehearsal flow and screenshot checklist, see [docs/api/DEMO_GUIDE.md](docs/api/DEMO_GUIDE.md).
