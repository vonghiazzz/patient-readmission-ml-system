# Demo Guide — Huy Final CatBoost

1. Start the API and open `/docs`.
2. Show `/ready`: model `huy-catboost-1.0.0`, feature set `HUY_FINAL_52`.
3. Expand the single `/predict` operation and point out 40 raw fields.
4. Submit `docs/api/sample_request.json`; expected prediction is 0.
5. Submit `docs/api/sample_high_risk_request.json`; expected prediction is 1.
6. Explain that the threshold is `0.8564852152742759` and the score is uncalibrated.
7. Show `models/production_huy/reports/evaluation_metrics.json`, SHAP and subgroup reports.
8. Show an invalid request with `patient_nbr`; expected response is structured HTTP 422.
9. Show `/metrics` without patient fields.

Before presenting:

```bash
python -m pytest -q
python scripts/smoke_test.py
docker compose config
```

Do not claim clinical readiness. State the holdout recall of 5.13% and the notebook leakage risks.
