# Contributing Guide

## Team Roles

| Member | Role            | Responsibilities                                                      |
| ------ | --------------- | --------------------------------------------------------------------- |
| Huy    | Data Lead       | Ingestion, validation, preprocessing, feature engineering, data tests |
| Khanh  | ML Lead         | Baselines, XGBoost, Optuna, MLflow, calibration, SHAP                 |
| Nghia  | Backend Lead    | FastAPI, schemas, Dockerfile, integration tests                       |
| Binh   | MLOps/Docs Lead | Docker Compose, Prometheus/Grafana, CI/CD, docs, slides               |

## Branching

main ← develop ← feature

- Each task requires a Pull Request (PR) and approval from at least one reviewer before merging into the `develop` branch.
- Feature development is frozen after Day 12; only critical bug fixes are permitted.

## Rules

- Do not commit raw patient data, .env files, or large model binaries.
- Each technical task must have a deliverable and evidence of a PR or commit.
