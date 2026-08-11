# Patient Readmission API Contract

## Authoritative contract

`models/production_v1/feature_manifest.json` is the source of truth for field names/order and
`models/production_v1/metadata.json` is the source of truth for model version and threshold. The
Pydantic model is generated from the manifest plus fitted preprocessor type semantics, so startup
fails or readiness is false when these artifacts disagree.

## Prediction

`POST /predict` accepts exactly 42 required source fields. `POST /api/v1/predict` is a compatibility
alias. Extra fields are forbidden. A categorical field may be explicitly `null` and is then handled
by the frozen imputer; numeric/count fields may not be null.

- Non-negative integer counts: `num_lab_procedures`, `num_procedures`, `num_medications`,
  `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`.
- `time_in_hospital`: integer of at least 1.
- Categorical integer IDs: `admission_type_id`, `admission_source_id`. They remain integer inputs
  but the frozen preprocessor routes them through its categorical encoder.
- All other manifest source features: required nullable categorical strings.

The service creates only `has_outpatient_history`, `has_emergency_history`, and
`has_inpatient_history`. Clients cannot send derived features. Identifiers, targets, weight,
discharge disposition, and diagnosis fields are excluded as defined by the manifest.

Inference is: 42 ordered source fields → three derived flags → 45 ordered V1 fields → frozen
preprocessor → 223 transformed features → frozen XGBoost raw probability → metadata threshold.

### Response

```json
{
  "model_version": "1.0.0",
  "risk_score": 0.23,
  "decision_threshold": 0.17,
  "prediction": 1,
  "status": "high_risk"
}
```

`prediction` uses an inclusive comparison: `risk_score >= decision_threshold`. The status is
`high_risk` for 1 and `not_high_risk` for 0. No undocumented risk bands are produced.

## Error contract

Errors never echo submitted values:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed.",
    "details": [
      {
        "location": ["body", "number_inpatient"],
        "message": "Input should be greater than or equal to 0",
        "error_type": "greater_than_equal"
      }
    ]
  }
}
```

| HTTP status | Meaning |
| --- | --- |
| `422` | Missing, invalid, or forbidden request field |
| `500` | Privacy-safe unexpected internal error |
| `503` | Frozen artifacts are unavailable or fail contract validation |

## Operations

- `GET /health`: process liveness only.
- `GET /ready`: 200 only when all artifacts load and cross-artifact validation passes; otherwise
  503 with `not_ready`.
- `GET /docs`, `/redoc`, `/openapi.json`: generated API documentation.
- `GET /metrics`: privacy-safe Prometheus exposition; no request values or demographic labels.
