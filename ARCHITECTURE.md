# Patient Readmission System Architecture

## Online inference path

```text
JSON request
  → FastAPI /predict
  → manifest-generated Pydantic validation (42 source fields)
  → three deterministic history flags
  → 45 ordered V1 model inputs
  → frozen scikit-learn preprocessor
  → 223 transformed features
  → frozen XGBoost predict_proba
  → threshold from metadata.json (0.17)
  → model_version 1.0.0 response
```

`feature_manifest.json` owns field names and order. `metadata.json` owns model version and decision
threshold. The service does not retrain, calibrate, tune, or select a model during build or startup.
All four production artifacts are validated together before readiness becomes true.

The response score is raw XGBoost `predict_proba`, not a post-hoc calibrated probability. Binary
prediction uses the inclusive rule `risk_score >= decision_threshold`. `status` is only a readable
copy of that binary outcome; the system defines no additional risk bands.

## Runtime components

```text
                         ┌──────────────┐
client ────────────────► │ FastAPI API  │
                         │ /predict     │
                         │ /ready       │
                         │ /metrics     │
                         └──────┬───────┘
                                │ scrape
                         ┌──────▼───────┐
                         │ Prometheus   │──── alert rules
                         └──────┬───────┘
                                │ datasource
                         ┌──────▼───────┐
                         │ Grafana      │
                         └──────────────┘

frozen artifacts ─► MLflow init logger ─► MLflow tracking server + persistent volume
```

MLflow is an offline tracking/evidence component. It is not called by `/predict` and its outage does
not change inference. The idempotent init service logs the already-frozen champion, metadata,
metrics, four artifacts, SHAP reports, and fairness reports without fitting a model.

## Artifact contract

| Artifact | Responsibility |
| --- | --- |
| `model.joblib` | Frozen XGBoost champion, best iteration 493 |
| `preprocessor.joblib` | Frozen 45-to-223 transformation |
| `feature_manifest.json` | Request, derived, model-input, and excluded fields |
| `metadata.json` | Version, threshold, evaluation metrics, selection decision |

The artifact directory may be overridden with `PRODUCTION_ARTIFACT_DIR` for safe failure testing.
There is no mock or fallback model. Missing or inconsistent artifacts produce `/ready` 503 and make
prediction unavailable.

## Monitoring and privacy

Prometheus records bounded labels only: HTTP method, route template, status code, binary prediction,
model version, and feature-set version. It does not label metrics with patient identifiers,
demographics, specialties, payer codes, or raw values. The API does not log request bodies.

The provisioned dashboard shows request volume, p95 latency, error rate, score distribution,
predicted-positive rate, and readiness. Alerts cover scrape/API unavailability, model not-ready,
elevated error rate, and high p95 latency.

## Persistence and boundaries

Docker Compose uses named volumes for MLflow, Prometheus, and Grafana state. Production model files
are copied into the immutable API image and are not mounted from raw-data directories. Raw and
interim patient-level datasets are not part of the image or Compose stack.

The service is a research prototype for prioritization and human review. It is not a diagnosis
system, certified medical device, or autonomous clinical decision-maker. See `MODEL_CARD.md` and
`RESPONSIBLE_AI.md` for evaluation and governance limits.
