# Responsible AI — Huy Final CatBoost

The API returns a relative 30-day readmission score for a course demonstration. A score or binary
prediction is not a diagnosis, severity rating or instruction to treat a patient.

## Human oversight

- Never deny care or automate treatment from this output.
- Confirm input accuracy and review broader clinical context.
- Do not present raw `risk_score` as a calibrated probability.
- Provide a non-model fallback when the service is unavailable.

## Measured limitations

At threshold `0.8564852152742759`, final holdout recall is only 0.0513. The model missed 907 of 956
positive encounters. The Brier score is 0.3856 and no post-hoc calibration model exists.

The preprocessing and SMOTENC methodology reproduce the notebook, including known leakage risks.
The reported metrics therefore must not be generalized beyond this experiment.

## Explainability and subgroup audit

`models/production_huy/reports/global_shap_huy_features.csv` contains native CatBoost SHAP global
importance. SHAP describes model behavior, not causality.

Subgroup reports cover race, gender and age. Small groups or groups with fewer than 30 positives are
flagged. Metric differences are descriptive and do not establish fairness.

## Privacy and monitoring

Requests are not logged by the application. Patient and encounter identifiers are forbidden by the
schema. Prometheus labels are bounded to routes, status codes, model version and binary outcomes.
