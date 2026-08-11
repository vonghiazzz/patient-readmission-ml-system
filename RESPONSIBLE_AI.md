# Responsible AI: Patient Readmission Prediction

## Decision-support boundary

Version 1.0.0 estimates relative 30-day readmission risk for prioritization and additional human
review. It must not diagnose, prescribe treatment, deny care, make automatic discharge decisions,
or replace a clinician. The 0.17 threshold is a frozen validation-derived operating point rather
than a clinical truth.

## Fairness interpretation

The subgroup report is a descriptive audit of race, gender, and age at the frozen threshold. It
reports sample size, positives, prevalence, PR-AUC, ROC-AUC, Brier score, precision, recall/TPR,
FPR, specificity, selection rate, and confusion counts. Slices with fewer than 200 rows or 30
positives receive a small-sample caution; this is a stability warning, not a formal fairness rule.

Observed female/male recall and FPR are similar in the available holdout, but metric proximity does
not establish fairness. Race results include lower recall for missing race, and age results vary more
substantially. Small race/age slices are especially uncertain. Differences can reflect prevalence,
missingness, historical access, documentation patterns, treatment pathways, or model behavior; the
audit does not identify causality.

Before real clinical use, governance owners should:

1. Define the relevant harms, benefits, protected groups, and acceptable operating trade-offs.
2. Validate prospectively and independently at each deployment site.
3. Review inclusion of race, gender, and age, including legal, ethical, and clinical rationale.
4. Test intersectional groups when sample sizes support meaningful estimates.
5. Provide an appeal/escalation path and preserve clinician authority.
6. Monitor subgroup sample sizes and outcomes over time; avoid conclusions from unstable slices.

Fairness analysis must not trigger automatic retraining or threshold changes. Any change requires a
new governed model version and validation process.

## Explainability

Global SHAP explains how the fitted model distributes contribution across features in the analyzed
sample. One-hot columns are grouped per patient before taking absolute values and averaging. SHAP
does not prove causal effects, treatment benefit, individual necessity, or the reason a patient will
be readmitted. It should support model review, not serve as a standalone clinical explanation.

## Data, privacy, and security

`patient_nbr` and `encounter_id` are excluded from model input and API responses. Do not record raw
requests, patient identifiers, diagnosis text, or PHI/PII in application logs, MLflow tags, metrics,
or monitoring labels. Raw patient-level datasets must remain outside Git. Access to artifacts,
evaluation data, reports, and inference logs should follow least-privilege and retention policies.

## Known limitations

- Positive prevalence is approximately 11%, and precision is low at the selected threshold.
- Validation ROC-AUC is about 0.6477; final holdout ROC-AUC about 0.6315 and PR-AUC about 0.1888.
- The fixed validation set was repeatedly used during model comparison.
- There is no external-hospital or prospective clinical validation.
- Raw XGBoost probabilities have no post-hoc calibration model.
- Data and clinical workflow drift may invalidate performance.
- Missing demographic values and small groups make some subgroup estimates unstable.
- This is a production-like research prototype, not a certified medical device.

## Monitoring and incident response

Monitor schema failures, feature missingness, unknown categories, probability/selection-rate drift,
performance after labels mature, subgroup gaps with confidence/sample-size context, latency, and
artifact checksums/version. A substantial drift, data-contract violation, unexplained subgroup harm,
or performance regression should make the service not-ready or trigger human review—not silent
fallback, automatic retraining, or an unreviewed threshold adjustment.
