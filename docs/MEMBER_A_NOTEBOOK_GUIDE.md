# Huy Notebook Production Guide

The authoritative model-development source is
`notebooks/reference/Huy-prediction-on-hospital-readmission.ipynb`.

Production code intentionally reproduces its final story:

- filtered cohort and first encounter per patient;
- `<30` positive target;
- manual admission, medication, age and diagnosis mappings;
- utilization/log/interaction features;
- 52-feature native CatBoost contract;
- stratified 80/20 split;
- final CatBoost with SMOTENC, class weights `{0:1,1:10}`;
- decision threshold `0.8564852152742759`.

Known preprocessing leakage and resampling risks are preserved for artifact compatibility and must be
disclosed. A methodology correction requires retraining and a new model version; it must never be
silently mixed with this fitted artifact.
