# Patient Readmission ML System — Final CatBoost

The MLOps system predicts the likelihood of patient **readmission within 30 days**, utilizing exclusively the **Final CatBoost** model from Huy's notebook.

The project is built as an **end-to-end ML system**, comprising:

* Data validation and preprocessing
* Feature engineering
* Production model serving
* MLflow experiment/artifact tracking
* REST API using FastAPI
* Docker and Docker Compose
* Prometheus monitoring
* Grafana dashboard
* Automated testing
* GitHub Actions CI
* Explainability and subgroup fairness analysis

> **Important:** This system is designed for educational purposes and to assist in prioritizing patient monitoring. It does not replace clinical judgment and must not be used as an autonomous clinical decision system.

---

## 1. Project Overview

### Problem

Patient readmission is a healthcare problem in which the system needs to estimate the likelihood of a patient returning to the hospital within 30 days of a treatment episode.

### Solution

The system utilizes a production CatBoost model to:

1. Receive 40 raw encounter fields from the client.
2. Validate the request using Pydantic.
3. Execute the feature engineering pipeline.
4. Apply the frozen preprocessing state.
5. Generate 52 ordered model features.
6. Produce a `risk_score` using CatBoost's `predict_proba`.
7. Apply the production decision threshold.
8. Return the prediction and risk status via REST API.

---

## 2. Production Contract

| Item                    | Value                        |
| ----------------------- | ---------------------------- |
| Champion model          | Final CatBoost               |
| Model version           | `huy-catboost-1.0.0`         |
| Feature set             | `HUY_FINAL_52`               |
| Raw API fields          | 40                           |
| Model input features    | 52                           |
| Categorical features    | 7                            |
| Decision threshold      | `0.8564852152742759`         |
| Probability source      | Raw CatBoost `predict_proba` |
| Probability calibration | Not calibrated               |
| Target                  | 30-day readmission           |

### Target mapping

```text
readmitted == "<30"       → 1
readmitted in {">30","NO"} → 0
```

### Production artifact bundle

The authoritative production bundle is:

```text
models/production_huy/
├── model.pkl
├── preprocessing_state.json
├── feature_manifest.json
├── metadata.json
├── reference_predictions.json
└── reports/
```

The bundle contains the final fitted CatBoost model, frozen preprocessing state, feature contract, model metadata, reference prediction cases and Huy-specific evaluation reports.

---

## 3. System Architecture

The detailed system architecture is documented in:

[`ARCHITECTURE.md`](ARCHITECTURE.md)

The system is organized into four main layers:

```text
Client / API
      ↓
ML Inference
      ↓
MLOps / Observability
      ↓
CI / Containerization
```

### Main runtime components

```text
FastAPI
MLflow
Prometheus
Grafana
Docker Compose
GitHub Actions
```

---

## 4. Repository Structure

```text
patient-readmission-ml-system/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│
├── docs/
│   ├── api/
│   └── architecture/
│
├── models/
│   └── production_huy/
│
├── monitoring/
│   ├── alert_rules.yml
│   ├── prometheus.yml
│   └── grafana/
│
├── scripts/
│
├── src/
│   ├── api/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   └── monitoring/
│
├── tests/
│   ├── data/
│   ├── integration/
│   ├── production/
│   └── unit/
│
├── ARCHITECTURE.md
├── CONTRIBUTING.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 5. Prerequisites

### Required

* Python 3.11
* Docker Desktop
* Git
* Git LFS

### Recommended local environment

Create a Python 3.11 virtual environment:

```bash
python -m venv .venv
```

Activate on Git Bash / Linux / macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Verify:

```bash
python --version
```

---

## 6. Installation

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify dependencies:

```bash
python -m pip check
```

Expected:

```text
No broken requirements found.
```

---

## 7. Run the API Locally Without Docker

Start FastAPI:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

Health:

```text
http://127.0.0.1:8000/health
```

Readiness:

```text
http://127.0.0.1:8000/ready
```

Metrics:

```text
http://127.0.0.1:8000/metrics
```

---

## 8. API Usage

### 8.1 Normal prediction

The repository contains:

```text
docs/api/sample_request.json
```

Call:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @docs/api/sample_request.json
```

Example production response:

```json
{
  "model_version": "huy-catboost-1.0.0",
  "risk_score": 0.6643320154788986,
  "decision_threshold": 0.8564852152742759,
  "prediction": 0,
  "status": "not_high_risk"
}
```

### 8.2 High-risk prediction

The repository also contains:

```text
docs/api/sample_high_risk_request.json
```

Call:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @docs/api/sample_high_risk_request.json
```

Example high-risk response:

```json
{
  "model_version": "huy-catboost-1.0.0",
  "risk_score": 0.9379868301418867,
  "decision_threshold": 0.8564852152742759,
  "prediction": 1,
  "status": "high_risk"
}
```

### 8.3 Request contract

The API accepts the **40 raw encounter fields** required by the production contract.

Clients must not submit the 52 engineered model features.

The backend performs:

* medication encoding
* diagnosis grouping
* utilization features
* medication-count features
* interaction features
* log transformations
* standardization
* min-max transformations

Object key ordering in JSON does not affect inference. Field names, types and values must satisfy the Pydantic schema.

---

## 9. Docker Compose

Docker Compose is the recommended way to run the full local MLOps stack.

### Build

```bash
docker compose build
```

### Start

```bash
docker compose up -d
```

### Check status

```bash
docker compose ps
```

Expected services:

```text
api
mlflow
mlflow-init
prometheus
grafana
```

The main runtime services are exposed on:

| Service    |   Port | URL                   |
| ---------- | -----: | --------------------- |
| FastAPI    | `8000` | http://localhost:8000 |
| MLflow     | `5050` | http://localhost:5050 |
| Prometheus | `9090` | http://localhost:9090 |
| Grafana    | `3000` | http://localhost:3000 |

### Stop

```bash
docker compose down
```

To remove persistent volumes as well:

```bash
docker compose down -v
```

---

## 10. MLflow

MLflow is used to track the production model run and artifacts.

Open:

```text
http://localhost:5050
```

Experiment:

```text
patient-readmission-production
```

Production run:

```text
huy-catboost-production-1.0.0
```

The run records:

* parameters
* evaluation metrics
* production tags
* model version
* feature set
* artifacts

### Tracked production artifacts

```text
model.pkl
preprocessing_state.json
feature_manifest.json
metadata.json
reference_predictions.json
```

The API does not retrain the model during startup or inference.

---

## 11. Monitoring

### Prometheus

Open:

```text
http://localhost:9090
```

Prometheus scrapes:

```text
api:8000/metrics
```

The current monitoring layer includes metrics for:

* HTTP request count
* HTTP request latency
* HTTP errors
* model readiness
* prediction count
* prediction risk-score distribution

### Grafana

Open:

```text
http://localhost:3000
```

The current dashboard provides:

* Request volume
* Request latency
* Error rate
* Prediction-risk distribution
* Predicted-positive rate
* Model readiness

### Generate prediction traffic

Use the sample requests to populate dashboard metrics:

```bash
for i in {1..20}; do
  curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    --data @docs/api/sample_request.json > /dev/null
done
```

And high-risk traffic:

```bash
for i in {1..20}; do
  curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    --data @docs/api/sample_high_risk_request.json > /dev/null
done
```

Refresh Grafana after generating traffic.

---

## 12. Prometheus Alert Rules

The repository contains Prometheus rules for:

* API unavailable
* model not ready
* elevated API error rate
* high API latency

Check active rules:

```text
http://localhost:9090/alerts
```

Alert notification delivery through Alertmanager is not implemented in the current project.

---

## 13. Testing

### Ruff

```bash
python -m ruff check .
```

### Formatting

```bash
python -m ruff format --check .
```

### Unit/integration/production tests

```bash
python -m pytest -q
```

### Coverage

```bash
python -m pytest --cov=src --cov-report=term-missing -q
```

### Smoke test

```bash
python scripts/smoke_test.py
```

The repository includes tests covering API, data validation, splitting, model/evaluation contracts, fairness, monitoring and production artifacts.

---

## 14. GitHub Actions CI

The current GitHub Actions workflow performs:

1. Checkout repository with Git LFS
2. Verify Huy production bundle
3. Setup Python 3.11.15
4. Install dependencies
5. Run `pip check`
6. Run Ruff lint
7. Run Ruff format check
8. Run Pytest with coverage
9. Build the production Docker image

Workflow:

```text
.github/workflows/ci.yml
```

The current CI coverage gate is:

```text
60%
```

The current workflow provides continuous integration and Docker build verification. Automated production deployment is not implemented.

---

## 15. Model Evaluation

The reproduced final holdout metrics are:

| Metric      |          Value |
| ----------- | -------------: |
| PR-AUC      | `0.1081018905` |
| ROC-AUC     | `0.5667913495` |
| Precision   | `0.1585760518` |
| Recall      | `0.0512552301` |
| F1          | `0.0774703557` |
| Brier score | `0.3855983188` |

Confusion matrix:

|                 | Predicted Negative | Predicted Positive |
| --------------- | -----------------: | -----------------: |
| Actual Negative |             10,115 |                260 |
| Actual Positive |                907 |                 49 |

These results reproduce the saved notebook output.

---

## 16. Explainability

The project uses native CatBoost SHAP analysis.

Implementation:

```text
src/evaluation/explainability.py
```

The explainability analysis is used to understand which features contribute to model predictions.

The model should be treated as a decision-support component rather than an autonomous clinical decision-maker.

---

## 17. Fairness and Responsible AI

The project includes subgroup analysis for:

* race
* gender
* age

Implementation:

```text
src/evaluation/fairness.py
```

The fairness analysis is descriptive and does not automatically declare the model fair or unfair.

Small subgroup samples are flagged with caution indicators to avoid overinterpreting unstable estimates.

The project documents:

* fairness considerations
* model limitations
* explainability
* responsible-use constraints
* healthcare deployment considerations

Related documentation:

```text
MODEL_CARD.md
RESPONSIBLE_AI.md
```

---

## 18. Reproducibility

The production inference contract is reconstructed using versioned artifacts rather than re-fitting preprocessing at API startup.

Key reproducibility artifacts include:

```text
models/production_huy/
├── model.pkl
├── preprocessing_state.json
├── feature_manifest.json
├── metadata.json
└── reference_predictions.json
```

The offline split pipeline also records a split manifest containing:

* split strategy
* random state
* train/test ratios
* cohort row count
* row counts
* patient counts
* class counts
* positive rate
* patient identifier hashes

---

## 19. Troubleshooting

### Docker Engine is not running

Check:

```bash
docker info
```

Then verify Docker Desktop is running before executing:

```bash
docker compose up -d
```

### A service is unhealthy

Run:

```bash
docker compose ps
```

Then inspect logs:

```bash
docker compose logs api
docker compose logs mlflow
docker compose logs prometheus
docker compose logs grafana
```

### API is not ready

Open:

```text
http://localhost:8000/ready
```

If readiness fails, inspect the API log:

```bash
docker compose logs api
```

The API readiness check validates the production model identity, feature order, preprocessing state, version and threshold.

### Grafana shows `No data`

First generate prediction traffic:

```bash
for i in {1..20}; do
  curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    --data @docs/api/sample_request.json > /dev/null
done
```

For prediction-positive metrics, also use:

```text
docs/api/sample_high_risk_request.json
```

Then refresh the Grafana dashboard.

### MLflow experiment has no visible run

Check:

```bash
docker compose logs mlflow-init
```

The initialization job logs the production run into:

```text
patient-readmission-production
```

### Ruff fails locally

Run:

```bash
python -m ruff check .
```

and:

```bash
python -m ruff format --check .
```

Use the reported file and line to fix formatting/import issues before pushing.

---

## 20. Project Limitations

The current implementation has several documented limitations:

* The final CatBoost probability is not calibrated.
* Holdout recall is low at the selected operating threshold.
* The saved model reproduces the existing notebook protocol and its associated preprocessing/resampling limitations.
* Monitoring focuses on service and prediction metrics rather than a complete data-drift and concept-drift system.
* Alert notification delivery is not implemented through Alertmanager.
* Production deployment is not automated through CI/CD.
* Docker Compose is intended for local/reproducible multi-service execution rather than large-scale orchestration.
* The current API serves a single production CatBoost model.
* The system must not be used for autonomous clinical decisions.

---

## 21. Quick Start

For a reviewer who only needs to run the system:

```bash
git clone <repository-url>
cd patient-readmission-ml-system

docker compose build
docker compose up -d
docker compose ps
```

Then open:

```text
Swagger:
http://localhost:8000/docs

MLflow:
http://localhost:5050

Prometheus:
http://localhost:9090

Grafana:
http://localhost:3000
```

Test the API:

```bash
curl --fail-with-body -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @docs/api/sample_request.json
```

Verify the project:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python scripts/smoke_test.py
```

---

## 22. Related Documentation

* [`ARCHITECTURE.md`](ARCHITECTURE.md)
* [`CONTRIBUTING.md`](CONTRIBUTING.md)
* [`MODEL_CARD.md`](MODEL_CARD.md)
* [`RESPONSIBLE_AI.md`](RESPONSIBLE_AI.md)
* [`docs/api/DEMO_GUIDE.md`](docs/api/DEMO_GUIDE.md)
* [`docs/`](docs/)

---

## 23. Project Status

The current branch contains the final Huy CatBoost production contract and the supporting MLOps infrastructure.

Verified components include:

* FastAPI production inference
* Docker Compose runtime
* MLflow experiment/run/artifacts
* Prometheus metrics
* Grafana dashboard
* Prometheus alert rules
* GitHub Actions CI
* automated tests
* Responsible AI documentation
* architecture documentation

The remaining project work is primarily final documentation/evidence packaging, presentation preparation and submission QA.
