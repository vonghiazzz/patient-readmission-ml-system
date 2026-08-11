# Final Technical QA Evidence

Audit date: 2026-08-11

## Source and test gates

| Gate | Result |
|---|---|
| Python | 3.11.15 |
| `python -m pip check` | Passed; no broken requirements |
| `ruff check .` | Passed |
| `ruff format --check .` | Passed; 44 files already formatted |
| `pytest -q` | 46 passed, 2 warnings |
| Coverage | 64.22%, required floor 60% |
| `git diff --check` | Passed |
| Docker build | Passed |
| Standalone container smoke | Passed |
| Docker Compose smoke | Passed for the current working tree |

The two test warnings are:

1. FastAPI/Starlette TestClient deprecation in an installed dependency.
2. XGBoost cross-version pickle warning. The serialized booster identifies
   XGBoost 3.3.0, while the Python 3.11 runtime uses the newest compatible pinned
   version, XGBoost 3.2.0. Regression predictions pass, but portable model-format
   export requires the artifact owner and is prohibited by the frozen-artifact
   rule during this audit.

## Frozen artifact integrity

The following SHA-256 values were identical before and after implementation:

| Artifact | SHA-256 |
|---|---|
| `model.joblib` | `39969d49040dbff7d48ff4719df08af3eefbb0a8b20b92554858edce3d74834d` |
| `preprocessor.joblib` | `b2eb74447ff816537c9cf9168ea7a06446a4fe300388f99d182de0959b392c00` |
| `feature_manifest.json` | `a04f7b7aa0ed74a3c9d2ccd9dfdd9560e5cce1df04532ddc56409512531e386f` |
| `metadata.json` | `7037d8786a46059117a17ecb94365f09e4e0abc035ce5caaa16360e0cb53ab82` |

No training, Optuna, SMOTE, threshold selection, calibration, or artifact rewrite
was run.

## Observed inference contract

| Contract item | Observed value |
|---|---|
| Request fields | 42 |
| Derived features | 3 |
| Model input features | 45 |
| Transformed features | 223 |
| Model version | 1.0.0 |
| Threshold source | `metadata.json` |
| Decision threshold | 0.17 |
| Best iteration | 493 |

The runtime path is: strict Pydantic request contract -> canonical 42-column
order -> three deterministic derived features -> frozen preprocessor -> 223
transformed columns -> frozen XGBoost `predict_proba` -> metadata threshold.

## Reproducibility blocker

`models/production_v1/` is present and verified locally but is currently
untracked by Git. CI and a clean clone cannot build the production image until
the team chooses and implements one reviewed policy:

- track the four frozen artifacts with Git LFS, or
- download them from a durable, checksum-verified artifact store before test
  and build.

The CI workflow deliberately fails early when the bundle is absent. Raw data,
interim data, local `.env`, and local MLflow state remain excluded from Git.
