# Model Card: XGBoost V1 Optuna 1.0.0

## Overview

This model estimates relative risk of hospital readmission within 30 days. The binary target is
`readmitted_30d`, derived from the source readmission outcome. It is a production-like research
prototype, not a certified clinical decision system.

- Model: XGBoost V1 Optuna
- Version: 1.0.0
- Feature set: V1
- Best iteration: 493
- Probability: raw `predict_proba`; no post-hoc calibration
- Operating threshold: 0.17, selected on validation data and read at runtime from
  `models/production_v1/metadata.json`

## Intended use

Intended users are analysts and appropriately governed clinical operations teams. Appropriate uses
are risk prioritization, analytical decision support, and identifying cases for additional human
review. The score must be considered with clinical context and human judgment.

The model is not intended for diagnosis, autonomous clinical or treatment decisions, denial of
care, automatic discharge decisions, or replacement of clinicians.

## Data and evaluation

The project uses the UCI diabetes hospital encounter dataset. The positive class prevalence is
approximately 11%. Repository data preparation uses patient-aware splitting so a patient does not
cross its train/test boundary. The frozen model-selection workflow also used a fixed validation set
for repeated tuning and comparison; this may make validation performance optimistic. There has
been no external-hospital validation.

| Evaluation | PR-AUC | ROC-AUC | Brier |
| --- | ---: | ---: | ---: |
| Validation | 0.209404 | 0.647695 | 0.098366 |
| Final holdout | 0.188837 | 0.631469 | 0.099806 |

At threshold 0.17, final holdout precision is 0.182828, recall 0.387746, F1 0.248490, and
specificity 0.775515. The threshold is validation-derived, not a universal clinical cutoff.

## Features and leakage controls

The API accepts exactly 42 source features. It adds only three deterministic history indicators,
creating 45 ordered V1 inputs; the frozen preprocessor expands these to 223 transformed features.
`admission_type_id` and `admission_source_id` are integer-valued but processed categorically.

Identifiers and leakage-risk fields excluded from prediction include `encounter_id`, `patient_nbr`,
`readmitted`, `readmitted_30d`, `weight`, `discharge_disposition_id`, `diag_1`, `diag_2`, and
`diag_3`. Patient identifiers may be used for split/audit grouping only.

## Model selection

The frozen champion is the tuned V1 XGBoost. A soft-voting candidate had validation PR-AUC
0.210101, only +0.000521 mean paired patient-cluster bootstrap delta versus XGBoost, with 95% CI
[-0.001880, 0.002827]. The improvement was not robust enough to justify additional serving
complexity, so the ensemble was rejected. Model selection is closed and productionization does not
retrain or retune the champion.

## Explainability

Global SHAP outputs are stored under `models/production_v1/reports/`. One-hot contributions are
first summed per original feature for each patient; global importance is then the mean absolute
grouped contribution. This avoids inflating high-cardinality features. SHAP describes model behavior
and associations, not causes.

## Subgroup audit

The descriptive audit covers race, gender, and age using the frozen threshold. Female and male
recall were approximately 0.3932 and 0.3812, with FPR approximately 0.2241 and 0.2249. No large
gender difference was observed in this holdout at this threshold, but that does not prove fairness.
Race and age results vary more: for example, African American and Caucasian recall were about
0.4127 and 0.3818, while missing-race recall was about 0.1915. Small groups are flagged when
`n < 200` or positive cases `< 30`; their estimates are unstable.

## Limitations and risks

- Discrimination is moderate (holdout ROC-AUC about 0.6315; PR-AUC about 0.1888).
- Raw XGBoost probabilities are not post-hoc calibrated.
- Dataset age, coding conventions, missingness, and local workflows may not represent current or
  external hospitals.
- Repeated validation use, class imbalance, distribution shift, and subgroup sample size limit
  generalization.
- Race, gender, and age are predictors and require governance review before any real clinical use.
- Predictions and SHAP values are associations and must not be interpreted as causal or diagnostic.
- The joblib booster records XGBoost 3.3.0, while the Python 3.11 service must use the newest
  compatible runtime, XGBoost 3.2.0. Loading emits a cross-version pickle warning. Reproducibility
  is regression-tested, but native XGBoost format would be more portable in a future governed model
  version; the frozen 1.0.0 artifact is not regenerated during productionization.

## Privacy and monitoring

Never log or return patient/encounter identifiers, commit patient-level raw data, or place PHI/PII in
monitoring labels. Monitor input schema, missing/unknown-category rates, score and selection-rate
drift, delayed performance (PR-AUC, recall, precision, specificity, Brier), subgroup metrics with
sample sizes, latency, and artifact/model version. Revalidation and governance approval are required
before deployment to a new site or material workflow change.
