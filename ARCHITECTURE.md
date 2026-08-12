# Architecture — Huy Final CatBoost

## Online inference

```text
POST /predict
  → strict Pydantic validation of 40 raw encounter fields
  → Huy manual mappings and diagnosis grouping
  → utilization, medication-count and interaction features
  → frozen log/standard/min-max transformations
  → 52 ordered CatBoost features (7 categorical)
  → final CatBoost predict_proba[:, 1]
  → metadata threshold 0.8564852152742759
  → versioned JSON response
```

The model and JSON contract files under `models/production_huy/` load once during FastAPI lifespan.
`/ready` returns 200 only after model identity, embedded feature order, categorical indices,
preprocessing state, version and threshold validate together.

The public API never accepts identifiers, targets, secondary diagnoses or already-engineered model
features. The compatibility alias `/api/v1/predict` executes the same Huy model and is hidden from
Swagger; there is no second experimental model endpoint.

## Offline reconstruction

`src/data/splitting.py` applies Huy's row exclusions, keeps the first encounter per `patient_nbr`,
reproduces both outlier filters and performs the stratified 80/20 split. The fitted preprocessing
statistics reproduce the notebook and are never re-fitted by the API.

## Evaluation and observability

- `src/models/evaluate.py`: holdout metrics and calibration curve.
- `src/evaluation/explainability.py`: native CatBoost SHAP.
- `src/evaluation/fairness.py`: race, gender and age subgroup audit.
- `src/evaluation/mlflow_champion.py`: logs the frozen Huy bundle without retraining.
- Prometheus records request, latency, error, readiness and bounded model prediction metrics.

## Artifact boundary

| Artifact | Role |
| --- | --- |
| `model.pkl` | Final fitted CatBoost |
| `preprocessing_state.json` | Frozen Huy scaling state |
| `feature_manifest.json` | Raw/model/categorical order |
| `metadata.json` | Version, threshold, metrics, limitations |
| `reference_predictions.json` | Reload regression cases |
