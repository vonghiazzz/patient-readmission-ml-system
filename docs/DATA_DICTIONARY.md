# Data Dictionary — Huy Contract

Dataset: Diabetes 130-US Hospitals for Years 1999–2008, 101,766 raw encounters and 50 columns.

## Target and cohort

`readmitted == "<30"` is positive; `">30"` and `"NO"` are negative. Huy excludes rows with missing
race/diagnoses, invalid gender or discharge disposition 11, keeps the first encounter per patient and
applies two z-score outlier filters. Final cohort: 56,653 rows.

## Public request groups

- Demographics: `race`, `gender`, `age`.
- Admission: `admission_type_id`, `discharge_disposition_id`, `admission_source_id`.
- Utilization and encounter counts: hospital days, labs, procedures, medications, outpatient,
  emergency, inpatient and diagnosis counts.
- Labs: `max_glu_serum`, `A1Cresult`; not tested is represented by `None`, `Unknown` or JSON null.
- Medication states: 21 retained medication columns with `No`, `Steady`, `Up`, `Down`.
- Flags: `change`, `diabetesMed`.
- Primary diagnosis: `diag_1`, grouped into nine level-1 categories by the backend.

The complete ordered list and exclusions are in `models/production_huy/feature_manifest.json`.

## Derived transformations

The backend derives medication counts, utilization logs, nine interactions/log-interactions,
diagnosis category and applies frozen standard/min-max state. The model receives 52 features, seven
of which are categorical.
