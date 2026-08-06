# Understanding the Reference Notebook and Completing Member A's Handoff

## What the reference notebook contains

`notebooks/reference/prediction-on-hospital-readmission.ipynb` is useful as a
catalog of ideas, but it mixes data work and model work in one stateful
notebook. It should remain a reference rather than become the production
pipeline.

| Reference cells | Topic | Owner | Project implementation |
| --- | --- | --- | --- |
| 7–19 | Load data, inspect schema and missing values | Member A | `src/data/ingestion.py`, `src/data/validation.py` |
| 20–43 | Feature and target engineering | Member A with Member B review | `src/features/build_features.py`, `src/data/splitting.py` |
| 44–69 | Aggregate EDA | Member A | Data-understanding notebook and handoff report |
| 70–101 | Encoding, scaling and feature preparation | Member A | `src/features/preprocessing.py` |
| 102–130 | Training and model comparison | Member B | Model training modules and MLflow |

## Ideas that can be retained

- Map `<30` to the positive class and map `>30`/`NO` to the negative class.
- Report class imbalance before Member B selects metrics or weighting.
- Derive service utilization from prior outpatient, emergency and inpatient
  counts.
- Investigate age, hospital duration, medication count, diagnoses and prior
  utilization using aggregate statistics.
- Keep diagnosis grouping as a versioned feature experiment rather than an
  undocumented notebook mutation.

## Patterns that must not be copied

1. The reference notebook fits dummy encoding and standardization on the full
   dataset. That leaks information from evaluation data. The project fits its
   preprocessor on `train` only.
2. It drops repeated patients to force independence. The project retains useful
   encounters and prevents overlap through patient-aware splitting.
3. Several SMOTE sections resample before a fresh split or outside a
   cross-validation pipeline. Resampling belongs to Member B and must occur
   only inside training folds.
4. The test set is repeatedly used to compare models. In this project, Member B
   must use cross-validation or an inner validation split from `train`; `test`
   stays locked until final evaluation.
5. Manual category replacement and row-by-row diagnosis loops are difficult to
   reproduce. Feature rules must live in tested functions and versioned
   manifests.
6. Accuracy alone is misleading for the imbalanced target. Member B must report
   AUPRC, recall, precision, F1, AUROC and later calibration metrics.

## Run the clean Member A workflow

From the repository root, activate the Python environment and run:

```bash
python -m src.data.ingestion
python -m src.data.validation
python -m src.data.splitting
python -m src.features.preprocessing
python -m src.data.ml_handoff
```

The last command reads the generated train/test files and fitted preprocessor,
then creates:

- `reports/member_a_ml_handoff.json`: machine-readable contract for Member B.
- `reports/member_a_ml_handoff.md`: human-readable aggregate report.

It verifies:

- approved target mapping;
- zero patient overlap between train and test;
- exact feature-order agreement with preprocessing metadata;
- successful preprocessing of a training sample;
- artifact checksums;
- a policy that keeps test data locked.

No model is trained by this command and no patient-level row is written to the
reports.

## What Member A sends to Member B

1. Train/test locations and the patient-aware split manifest.
2. Fitted preprocessor plus its SHA-256 checksum.
3. Ordered raw feature contract and transformed feature count.
4. Target mapping and aggregate class distribution.
5. Missingness/data-quality summary.
6. Explicit instruction to tune using training folds only.

Member B then owns Dummy/Logistic/XGBoost training, imbalance experiments,
MLflow runs, tuning and model evaluation.
