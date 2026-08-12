# Contributing Guide

## Team Roles

| Member | Role            | Responsibilities                                                      |
| ------ | --------------- | --------------------------------------------------------------------- |
| Huy    | ML/Data Lead    | Authoritative notebook, CatBoost artifact, cohort and feature story   |
| Khanh  | Evaluation      | Holdout metrics, calibration, MLflow and SHAP                         |
| Nghia  | Backend Lead    | Raw-input preprocessing, FastAPI contract and integration tests      |
| Binh   | MLOps/Docs Lead | Docker Compose, Prometheus/Grafana, CI/CD, docs and slides            |

## Branching

main ← develop ← feature/*

- Mỗi task phải có PR, ít nhất 1 reviewer duyệt trước khi merge vào develop.
- Freeze feature development sau Day 12; chỉ sửa lỗi nghiêm trọng.

## Rules

- Không commit raw patient data, .env, model binary lớn.
- Mỗi task kỹ thuật phải có deliverable + PR/commit evidence.
