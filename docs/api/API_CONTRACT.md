# Huy CatBoost API Contract

`POST /predict` is the single documented prediction endpoint. `/api/v1/predict` is a hidden alias
that invokes the same model.

The request contains exactly 40 raw fields in `models/production_huy/feature_manifest.json`.
Required raw groups are demographics, three admission/discharge IDs, encounter counts, two lab
results, 21 retained medication states, change/diabetes-med flags and primary diagnosis `diag_1`.

`patient_nbr`, `encounter_id`, targets, `diag_2`, `diag_3`, payer code, medical specialty, weight,
examide, citoglipton and engineered features are forbidden.

The backend generates the 52-field ordered model input. JSON object key order is irrelevant; exact
field names and valid values are required.

Success response:

```json
{
  "model_version": "huy-catboost-1.0.0",
  "risk_score": 0.6643320154788986,
  "decision_threshold": 0.8564852152742759,
  "prediction": 0,
  "status": "not_high_risk"
}
```

`prediction = 1` exactly when `risk_score >= decision_threshold`. The score is raw and uncalibrated.

| Status | Meaning |
| --- | --- |
| 200 | Prediction returned |
| 422 | Missing, extra or invalid field |
| 500 | Unexpected internal error |
| 503 | Huy production bundle unavailable |

Use `docs/api/sample_request.json` for prediction 0 and
`docs/api/sample_high_risk_request.json` for prediction 1.
