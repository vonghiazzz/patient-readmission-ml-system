# Docker Compose Smoke-Test Evidence

Audit date: 2026-08-11
Environment: macOS host, OrbStack Docker 27.4.1, Python 3.11.15

## Result

The checked-out working tree passed the complete Compose integration test. This
is not evidence of a clean-clone run because `models/production_v1/` is currently
untracked; an artifact distribution policy must be completed before another
machine or CI runner can reproduce the build.

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
- The image contains only the four production contract artifacts plus approved
  reports; raw/interim datasets and the unrelated CatBoost file are excluded.
- Current image size is approximately 1.87 GB. This is functional but remains
  an optimization opportunity because the shared requirements contain heavy
  evaluation packages.

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
