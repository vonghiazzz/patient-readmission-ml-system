# Member A → Member B Data Handoff

Generated: `2026-08-06T16:23:01.945507+00:00`

## Dataset and target

- Rows: **101766**
- Columns: **50**
- Unique patients: **71518**
- Duplicate encounter IDs: **0**
- Positive rows (`<30`): **11357** (**11.16%**)
- Target mapping: `<30 → 1`, `>30 → 0`, `NO → 0`

## Patient-aware split

| Split | Rows | Patients | Positive rate |
| --- | ---: | ---: | ---: |
| Train | 71520 | 50062 | 11.22% |
| Test | 30246 | 21456 | 11.03% |

Patient overlap is verified as **zero**. Member B must use cross-validation or an inner validation split from `train` and keep `test` locked.

## Feature and preprocessing contract

- Raw model inputs: **49**
- Transformed features: **2331**
- Numeric inputs: **13**
- Categorical inputs: **36**
- Preprocessor version: `v1`
- Preprocessor SHA-256: `e22e630afbfe5d792802e88743fbad7ca5f22b648574bbba91eba6c1ffbc88e8`
- Excluded targets: `readmitted`, `readmitted_30d`
- Excluded identifiers: `encounter_id`, `patient_nbr`

## Highest missing rates

| Column | Missing rows | Missing rate |
| --- | ---: | ---: |
| `weight` | 98569 | 96.86% |
| `max_glu_serum` | 96420 | 94.75% |
| `a1cresult` | 84748 | 83.28% |
| `medical_specialty` | 49949 | 49.08% |
| `payer_code` | 40256 | 39.56% |
| `race` | 2273 | 2.23% |
| `diag_3` | 1423 | 1.40% |
| `diag_2` | 358 | 0.35% |
| `diag_1` | 21 | 0.02% |

## Handoff boundary

Member A owns ingestion, validation, patient-aware splitting, feature construction and the fitted preprocessing artifact. Member B owns resampling inside training folds, baseline/model training, MLflow tracking, tuning and evaluation.

This report contains aggregate statistics only and no patient-level records.
