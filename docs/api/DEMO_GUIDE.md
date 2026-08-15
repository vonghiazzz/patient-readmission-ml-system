# Patient Readmission ML System — Demo Guide

## 1. Demo Objective

This guide provides a reproducible live-demo sequence for the Patient Readmission ML System.

The demonstration covers:

1. Application startup
2. Swagger/OpenAPI API
3. Real CatBoost prediction
4. Prometheus metrics
5. Grafana monitoring
6. MLflow experiment tracking
7. Health/readiness verification
8. Backup evidence when live services are unavailable

The production model used in the demo is:

```text
huy-catboost-1.0.0
```

The production decision threshold is:

```text
0.8564852152742759
```

---

## 2. Demo Environment

Required:

* Docker Desktop
* Git
* Git LFS
* Repository checkout
* Python 3.11 for local test/verification commands

Recommended demo environment:

```text
Docker Compose
├── FastAPI
├── MLflow
├── Prometheus
└── Grafana
```

---

## 3. Start the System

From the project root:

```bash
docker compose up -d
```

Verify:

```bash
docker compose ps
```

All main services should report a healthy/running state.

Expected services:

```text
api
mlflow
mlflow-init
prometheus
grafana
```

Useful logs:

```bash
docker compose logs api
docker compose logs mlflow
docker compose logs prometheus
docker compose logs grafana
```

---

## 4. Demo Step 1 — Swagger / OpenAPI

Open:

```text
http://localhost:8000/docs
```

Show:

```text
GET  /health
GET  /ready
GET  /metrics
POST /predict
```

Explain:

> FastAPI provides the model-serving interface and automatically exposes the OpenAPI/Swagger documentation.

### Evidence

Capture a screenshot showing the Swagger UI and the production `/predict` endpoint.

Suggested filename:

```text
evidence_swagger.png
```

---

## 5. Demo Step 2 — Health and Readiness

Open:

```text
http://localhost:8000/health
```

Then:

```text
http://localhost:8000/ready
```

The `/ready` endpoint confirms that the production model contract has been loaded successfully.

The readiness check covers the production model identity and related serving metadata.

### Talking point

> `/health` indicates the API process is running, while `/ready` verifies that the production model is ready to serve predictions.

### Evidence

Suggested filename:

```text
evidence_api_ready.png
```

---

## 6. Demo Step 3 — Normal Prediction

The repository provides:

```text
docs/api/sample_request.json
```

Execute:

```bash
curl --fail-with-body -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @docs/api/sample_request.json
```

Example response:

```json
{
  "model_version": "huy-catboost-1.0.0",
  "risk_score": 0.6643320154788986,
  "decision_threshold": 0.8564852152742759,
  "prediction": 0,
  "status": "not_high_risk"
}
```

Explain:

```text
risk_score = 0.6643
threshold  = 0.8565
```

Therefore:

```text
0.6643 < 0.8565
→ prediction = 0
→ not_high_risk
```

### Talking point

> The API returns the raw model risk score together with the production decision threshold and the final prediction status.

---

## 7. Demo Step 4 — High-Risk Prediction

The repository also provides:

```text
docs/api/sample_high_risk_request.json
```

Execute:

```bash
curl --fail-with-body -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @docs/api/sample_high_risk_request.json
```

The expected result is a high-risk prediction:

```text
prediction = 1
status = high_risk
```

This request is also used to generate positive prediction metrics for the monitoring dashboard.

---

## 8. Demo Step 5 — Prometheus Metrics

Open:

```text
http://localhost:9090
```

The API exposes:

```text
http://localhost:8000/metrics
```

Prometheus scrapes this endpoint.

Useful metrics include:

```text
readmission_http_requests_total
readmission_http_request_duration_seconds
readmission_http_errors_total
readmission_model_ready
readmission_predictions_total
readmission_prediction_risk_score
```

Example PromQL:

```promql
readmission_predictions_total
```

For model readiness:

```promql
readmission_model_ready
```

For request latency:

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(readmission_http_request_duration_seconds_bucket[5m])
  )
)
```

### Talking point

> Prometheus collects both system-level API metrics and prediction-related metrics from the serving layer.

---

## 9. Demo Step 6 — Grafana Dashboard

Open:

```text
http://localhost:3000
```

Open the:

```text
Patient Readmission API
```

dashboard.

The dashboard should show:

* Request volume
* Request latency
* Error rate
* Prediction-risk distribution
* Predicted-positive rate
* Model readiness

### Generate traffic when needed

Normal traffic:

```bash
for i in {1..20}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    --data @docs/api/sample_request.json > /dev/null
done
```

High-risk traffic:

```bash
for i in {1..20}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    --data @docs/api/sample_high_risk_request.json > /dev/null
done
```

Refresh Grafana after generating traffic.

### Evidence

Suggested filename:

```text
evidence_grafana_monitoring.png
```

The screenshot should show real data rather than `No data` panels.

---

## 10. Demo Step 7 — MLflow

Open:

```text
http://localhost:5050
```

Open the experiment:

```text
patient-readmission-production
```

Then open the production run:

```text
huy-catboost-production-1.0.0
```

Show:

* Parameters
* Metrics
* Tags
* Artifacts

Expected production artifacts:

```text
model.pkl
preprocessing_state.json
feature_manifest.json
metadata.json
reference_predictions.json
```

### Talking point

> MLflow provides traceability for the final production model by storing the run metadata, parameters, metrics, tags and artifact bundle.

### Evidence

Suggested filenames:

```text
evidence_mlflow_run.png
evidence_mlflow_artifacts.png
```

---

## 11. Demo Step 8 — CI

Open the GitHub repository:

```text
Actions
```

Show the latest successful:

```text
CI / quality
```

The pipeline verifies:

1. Production model bundle
2. Python environment
3. Dependencies
4. Ruff lint
5. Ruff format
6. Pytest
7. Coverage
8. Docker build

### Talking point

> GitHub Actions prevents broken code and invalid production artifacts from progressing through the repository workflow.

### Evidence

Suggested filename:

```text
evidence_github_actions_ci.png
```

---

## 12. Demo Step 9 — Repository Architecture

Open:

```text
ARCHITECTURE.md
```

Show the high-level Mermaid architecture and briefly explain:

```text
Client
  ↓
FastAPI
  ↓
Feature Pipeline
  ↓
CatBoost
  ↓
Prediction
  ↓
Prometheus / Grafana

MLflow
  ↕
Production artifact tracking

GitHub Actions
  ↓
Quality checks + Docker build
```

### Talking point

> The architecture separates online inference from offline model/evaluation workflows while keeping model artifacts, monitoring and CI traceable.

---

## 13. Suggested Live Demo Order

For a 5–7 minute demo:

### 1. Start services

```bash
docker compose up -d
docker compose ps
```

### 2. Swagger

```text
http://localhost:8000/docs
```

### 3. Normal prediction

```text
sample_request.json
```

### 4. High-risk prediction

```text
sample_high_risk_request.json
```

### 5. Grafana

Show real request and prediction metrics.

### 6. MLflow

Show production run and artifacts.

### 7. GitHub Actions

Show successful CI.

### 8. Finish

Explain:

* reproducibility
* observability
* artifact traceability
* limitations

---

## 14. Backup Demo Plan

If Docker or internet access fails during the presentation:

### Backup 1 — Swagger screenshots

Use:

```text
evidence_swagger.png
evidence_api_ready.png
```

### Backup 2 — Prediction screenshots

Use the saved `/predict` request/response screenshots.

### Backup 3 — Grafana screenshot

Use:

```text
evidence_grafana_monitoring.png
```

### Backup 4 — MLflow screenshots

Use:

```text
evidence_mlflow_run.png
evidence_mlflow_artifacts.png
```

### Backup 5 — GitHub Actions

Use:

```text
evidence_github_actions_ci.png
```

The backup demo should still communicate the complete end-to-end workflow even when live services cannot be started.

---

## 15. Troubleshooting During Demo

### API is not ready

```bash
docker compose ps
docker compose logs api
```

Check:

```text
http://localhost:8000/ready
```

### Grafana shows no data

Generate prediction traffic again using the two sample JSON files and refresh the dashboard.

### MLflow has no visible run

Check:

```bash
docker compose logs mlflow-init
```

### Prometheus has no metrics

Check:

```text
http://localhost:8000/metrics
```

Then verify the Prometheus target.

### Docker service is unhealthy

Run:

```bash
docker compose ps
docker compose logs <service-name>
```

---

## 16. Evidence Checklist

Before presentation, confirm the following files exist:

```text
evidence_swagger.png
evidence_api_ready.png
evidence_grafana_monitoring.png
evidence_mlflow_run.png
evidence_mlflow_artifacts.png
evidence_github_actions_ci.png
```

Also keep the following screenshots available as backup:

```text
architecture_overview.png
predict_normal_response.png
predict_high_risk_response.png
```

---

## 17. Final Demo Checklist

* [ ] Docker Desktop is running
* [ ] `docker compose up -d` succeeds
* [ ] All services are healthy
* [ ] Swagger loads
* [ ] `/ready` returns successfully
* [ ] Normal `/predict` request works
* [ ] High-risk `/predict` request works
* [ ] Prometheus receives metrics
* [ ] Grafana dashboard contains real data
* [ ] MLflow production run is visible
* [ ] MLflow artifacts are visible
* [ ] GitHub Actions CI is green
* [ ] Backup screenshots are ready
* [ ] Demo fits the planned presentation time
