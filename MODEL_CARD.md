# Model Card: Huy Final CatBoost

## Identity

- Model: `CatBoostClassifier`
- Version: `huy-catboost-1.0.0`
- Model SHA-256: `a0d11b7ed0c1956d10afbfda360ec24ae2c55f6d6d50d32ed50780a81160331b`
- Positive class: readmission within 30 days (`readmitted == "<30"`).
- Operating threshold: `0.8564852152742759`.
- Output: raw, uncalibrated `predict_proba`.

## Cohort and training

The raw 101,766 encounters are filtered for missing diagnoses/race, invalid gender and discharge
disposition 11. The first encounter per patient is retained and two notebook outlier filters leave
56,653 rows. The final split is stratified 80/20 with `random_state=42`.

The final model uses SMOTENC on the training split with `sampling_strategy=0.7`, followed by CatBoost
with 500 trees, depth 4, learning rate 0.018489688756468402 and class weights `{0:1,1:10}`.

## Holdout metrics

| Metric | Value |
| --- | ---: |
| PR-AUC | 0.10810189054977758 |
| ROC-AUC | 0.566791349498412 |
| Precision | 0.15857605177993528 |
| Recall | 0.051255230125523014 |
| F1 | 0.0774703557312253 |
| Brier | 0.38559831881602613 |

At the operating threshold: TN 10,115; FP 260; FN 907; TP 49.

## Intended use

Course demonstration and governed exploration of readmission prioritization. It is not suitable for
diagnosis, treatment selection, denial of care, autonomous intervention or unsupervised deployment.

## Limitations

- Final recall is 5.13%, so most positives in the holdout are missed.
- The raw probability is not calibrated and must not be interpreted as clinical certainty.
- Notebook preprocessing statistics were fit before the split, creating evaluation leakage.
- SMOTENC was performed before final CV rather than inside each fold.
- Thresholds and subgroup results are dataset-specific.
- Medication direction is collapsed: `Steady`, `Up` and `Down` all become medication-used = 1.

Huy-specific SHAP, calibration and subgroup files are under `models/production_huy/reports/`.
