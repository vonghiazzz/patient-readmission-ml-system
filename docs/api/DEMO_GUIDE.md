# Backend and MLOps Demo Guide

All request data in this guide is synthetic. Do not use screenshots containing identifiers, raw
patient records, tokens, credentials, filesystem paths, or terminal environment dumps.

## 1. Start the stack

```bash
cp .env.example .env
docker compose config
docker compose up --build -d
```

Wait for `api`, `mlflow`, `prometheus`, and `grafana` to become healthy. `mlflow-init` is a one-shot
service and should finish with exit code 0.

```bash
docker compose ps
docker compose logs --tail=100 mlflow-init
```

## 2. Health and readiness

```bash
curl --fail http://localhost:8000/health | python -m json.tool
curl --fail http://localhost:8000/ready | python -m json.tool
```

Explain that health is process liveness, while readiness requires all four frozen artifacts and the
42 → 3 → 45 → 223 contract. Readiness must show model version `1.0.0` and feature set `V1`.

## 3. Swagger

Open <http://localhost:8000/docs>. Swagger must show both `POST /predict` and
`POST /predict/catboost`. Expand `/predict` to show 42 XGBoost source fields, then expand
`/predict/catboost` to show 52 already-engineered CatBoost fields. Both examples are synthetic.

## 4. Valid prediction

```bash
curl --fail-with-body -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data @docs/api/sample_request.json | tee /tmp/readmission-prediction.json
python -m json.tool /tmp/readmission-prediction.json
```

Explain the response precisely:

- `risk_score` is the raw frozen XGBoost `predict_proba` output, not calibrated clinical certainty.
- `decision_threshold` is `0.17` from `metadata.json`, not a universal clinical cutoff.
- `prediction` is 1 when `risk_score >= decision_threshold`, otherwise 0.
- `status` mirrors the binary prediction; no additional risk-band policy exists.
- `model_version` is the model contract version. It is distinct from the API/service version even if
  both currently display `1.0.0`.

Run the experimental CatBoost request separately:

```bash
curl --fail-with-body -X POST http://localhost:8000/predict/catboost \
  -H 'Content-Type: application/json' \
  --data @docs/api/sample_catboost_request.json
```

Explain that CatBoost uses its embedded threshold `0.5` and is identified by model type plus
artifact SHA-256 because it has no approved semantic version. It does not replace XGBoost V1.

## 5. Invalid prediction

```bash
curl -i -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data '{"patient_nbr":"must-not-be-accepted"}'
```

Show HTTP 422 and the stable validation error. Confirm the submitted value is not echoed.

## 6. Automated API smoke

```bash
python scripts/smoke_test.py --base-url http://localhost:8000
```

This checks `/health`, `/ready`, XGBoost, CatBoost, the compatibility alias, invalid input, and
`/metrics`.

## 7. Prometheus metrics and target

```bash
curl --fail http://localhost:8000/metrics | grep '^readmission_'
curl --fail 'http://localhost:9090/api/v1/targets' | python -m json.tool
```

Open <http://localhost:9090/targets> and show job `readmission-api` as `UP`. Open
<http://localhost:9090/alerts> and show all four rules loaded. Metric labels must contain no patient,
encounter, demographic, specialty, payer, or raw input values.

## 8. Grafana

Open <http://localhost:3000>, sign in with the local credentials from `.env`, and open folder
**Patient Readmission** → dashboard **Patient Readmission API**. Show:

1. request volume;
2. p95 request latency;
3. error rate;
4. prediction-risk distribution;
5. predicted-positive rate;
6. model readiness.

The automatically provisioned datasource must report success and use `http://prometheus:9090`.

## 9. MLflow champion

Open <http://localhost:5050>, experiment `patient-readmission-production`, run
`xgb-v1-production-1.0.0`. Show version, feature set, best iteration, threshold, evaluation metrics,
four production artifacts, and SHAP/fairness reports. The logger only reloads the frozen files; it
does not train.

Verify persistence without deleting volumes:

```bash
docker compose restart mlflow
```

Reload MLflow and confirm the run remains visible.

## 10. Safe failure demo

Do not rename or delete artifacts. Start the same image on port 8001 with a missing configured path:

```bash
docker run --rm -p 8001:8000 \
  -e PRODUCTION_ARTIFACT_DIR=/app/models/unavailable \
  patient-readmission-api:1.0.0
```

In another terminal:

```bash
curl -i http://localhost:8001/health
curl -i http://localhost:8001/ready
curl -i -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  --data @docs/api/sample_request.json
```

Expected: health 200, readiness 503, prediction 503, no stack trace or internal path.

## 11. Stop cleanly

```bash
docker compose down
```

Do not add `-v` during normal rehearsal because that deletes monitoring and MLflow evidence.

## Evidence checklist

- [ ] `docker compose ps` with required services healthy and `mlflow-init` exited 0
- [ ] `/health` response
- [ ] `/ready` response showing `1.0.0` and `V1`
- [ ] Swagger `/predict` 42-field example
- [ ] Swagger `/predict/catboost` 52-field example
- [ ] successful synthetic prediction response
- [ ] invalid request 422 without submitted value
- [ ] `/metrics` privacy-safe metric names
- [ ] Prometheus target `UP`
- [ ] Prometheus alert rules loaded
- [ ] Grafana dashboard with four required panels populated
- [ ] Grafana datasource health success
- [ ] MLflow champion run, metrics, parameters, artifacts, and reports
- [ ] passing Ruff, pytest, coverage, and artifact-contract output
- [ ] safe unavailable-artifact demo

Capture screenshots only after a clean rehearsal and place approved evidence outside raw-data and
secret-bearing directories. Do not fabricate evidence when Docker or a service was not verified.
