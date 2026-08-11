# Docker Compose Smoke-Test Evidence

Audit date: 2026-08-11
Environment: macOS host, OrbStack Docker 27.4.1, Python 3.11.15

## Result

The checked-out working tree passed the complete Compose integration test. All required endpoint
artifacts are now tracked in Git and checked by CI before Docker build.

| Check | Observed result |
|---|---|
| `docker compose config --quiet` | Passed |
| API | Healthy on host port 8000 |
| Prometheus | Healthy on host port 9090 |
| Grafana | Healthy on host port 3000 |
| MLflow | Healthy on host port 5050 |
| `mlflow-init` | Exited 0 |
| API smoke | Health, readiness, valid prediction, alias, invalid request, and metrics passed |
| Prometheus target | `readmission-api`, `health=up`, no scrape error |
| Alert rules | Four rules loaded; all `health=ok` |
| Grafana datasource | Prometheus query succeeded, status `OK` |
| Grafana dashboard | `patient-readmission-api`, six required panels loaded |
| MLflow persistence | Same single champion run remained after MLflow restart |

Final healthy service snapshot:

- `api`: healthy
- `prometheus`: healthy
- `grafana`: healthy
- `mlflow`: healthy
- `mlflow-init`: exited 0

## API and monitoring observations

- Host smoke command: `python scripts/smoke_test.py --base-url http://127.0.0.1:8000`
- Prometheus target: `http://api:8000/metrics`, state `up`
- Alert rules: `ReadmissionApiUnavailable`, `ReadmissionModelNotReady`,
  `ReadmissionApiElevatedErrorRate`, and `ReadmissionApiHighLatency`
- Grafana panels: request volume, request latency, error rate,
  prediction-risk distribution, predicted-positive rate, and model readiness
- Metrics expose aggregate, bounded-cardinality labels only; no raw request or
  patient field is used as a label.

## MLflow observations

- Experiment: `patient-readmission-production`
- Run name: `xgb-v1-production-1.0.0`
- Run ID: `9cb5324944e64c04a5cabc9a293d7f97`
- Status: `FINISHED`
- Champion runs matching version/tag: exactly 1
- Parameters include `best_iteration=493`, `decision_threshold=0.17`, and the
  frozen Trial 22 hyperparameters.
- Ten validation/final metrics were present.
- Artifact copies: model, preprocessor, metadata, and feature manifest.
- Reports: global SHAP CSV, transformed SHAP plot, subgroup fairness report,
  and fairness gap summary.
- After `docker compose restart mlflow`, the same run ID and all eight files
  remained available from the named volume.

## Container observations

- API image ID: `sha256:df95002c207c9f5628f5f25187d0af1ad8b9867d1188c0cd7208f4b321e761f4`
- Runtime user: `appuser` (UID 10001 in the container)
- Healthcheck targets `/ready`.
- The image contains the four frozen XGBoost contract artifacts, the separately supplied CatBoost
  artifact, and approved reports. Raw/interim datasets are excluded.
- Current image size is approximately 2.23 GB. This is functional but remains
  an optimization opportunity because the shared requirements contain heavy
  evaluation packages.

## CatBoost endpoint extension

After the baseline Compose audit, the requested experimental CatBoost endpoint was added and
verified independently without modifying the four frozen XGBoost artifacts:

- Image ID: `sha256:009db36e8444d3c895e35b11980e86c2a943ea84290674655f8f4b3fba89d5b3`.
- `catboost==1.2.10` installed successfully in the Python 3.11.15 image.
- `cat_tunning_model.pkl` was copied into the image.
- Container healthcheck reached `healthy`.
- Host smoke test passed XGBoost, CatBoost, compatibility alias, invalid request, and metrics.
- `/predict/catboost` returned the artifact SHA-256 and threshold `0.5`.

## Failure-path evidence

Using `PRODUCTION_ARTIFACT_DIR=/app/models/unavailable` in a temporary container,
without deleting or renaming the production bundle:

- `/health` returned 200.
- `/ready` returned 503.
- `/predict` returned 503.
- The client response contained no stack trace, artifact path, or patient data.

An invalid request against the healthy stack returned structured 422.

## Shutdown

The final stack can be stopped with `docker compose down`. Named volumes should
be retained unless the operator intentionally wants to erase local monitoring
and MLflow history.
