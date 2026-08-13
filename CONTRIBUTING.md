# Contributing Guide

## Team Roles

| Member | Role            | Responsibilities                                                      |
| ------ | --------------- | --------------------------------------------------------------------- |
| Huy    | ML/Data Lead    | Authoritative notebook, CatBoost artifact, cohort and feature story   |
| Khanh  | Evaluation      | Holdout metrics, calibration, MLflow and SHAP                         |
| Nghia  | Backend Lead    | Raw-input preprocessing, FastAPI contract and integration tests      |
| Binh   | MLOps/Docs Lead | Docker Compose, Prometheus/Grafana, CI/CD, docs and slides            |

## Branching

main ← develop ← feature

- Each task requires a Pull Request (PR) and approval from at least one reviewer before merging into the `develop` branch.
- Feature development is frozen after Day 12; only critical bug fixes are permitted.

## Rules

- Do not commit raw patient data, .env files, or large model binaries.
- Each technical task must have a deliverable and evidence of a PR or commit.
