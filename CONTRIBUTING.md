# Contributing Guide

## Team Roles

| Member | Role            | Responsibilities                                                      |
| ------ | --------------- | --------------------------------------------------------------------- |
| Huy    | Data Lead       | Ingestion, validation, preprocessing, feature engineering, data tests |
| Khanh  | ML Lead         | Baselines, XGBoost, Optuna, MLflow, calibration, SHAP                 |
| Nghia  | Backend Lead    | FastAPI, schemas, Dockerfile, integration tests                       |
| Binh   | MLOps/Docs Lead | Docker Compose, Prometheus/Grafana, CI/CD, docs, slides               |

## Branching

main ← develop ← feature/*

- Mỗi task phải có PR, ít nhất 1 reviewer duyệt trước khi merge vào develop.
- Freeze feature development sau Day 12; chỉ sửa lỗi nghiêm trọng.

## Rules

- Không commit raw patient data, .env, model binary lớn.
- Mỗi task kỹ thuật phải có deliverable + PR/commit evidence.
