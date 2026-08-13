# Patient Readmission ML System — Final CatBoost

Hệ thống dự đoán nguy cơ tái nhập viện trong 30 ngày, sử dụng duy nhất mô hình CatBoost cuối cùng
trong notebook của Huy. Hệ thống phục vụ mục đích học tập và hỗ trợ ưu tiên theo dõi; không thay
thế quyết định lâm sàng.

## Production contract

| Item | Value |
| --- | --- |
| Champion | Final CatBoost |
| Model version | `huy-catboost-1.0.0` |
| Feature set | `HUY_FINAL_52` |
| Request | 40 raw encounter fields |
| Model input | 52 Huy-engineered features |
| Threshold | `0.8564852152742759` |
| Probability | Raw CatBoost `predict_proba`; not calibrated |
| Source notebook | `notebooks/reference/Huy-prediction-on-hospital-readmission.ipynb` |

Target:

```text
readmitted == "<30" → 1
readmitted in {">30", "NO"} → 0
```

The authoritative production bundle is `models/production_huy/`:

- `model.pkl`: fitted final CatBoost with `class_weights={0:1,1:10}`.
- `preprocessing_state.json`: frozen notebook scaler/min-max state.
- `feature_manifest.json`: 40-field raw and 52-field model contracts.
- `metadata.json`: version, threshold, training protocol, metrics and limitations.
- `reference_predictions.json`: reproducible prediction-0 and prediction-1 cases.
- `reports/`: Huy-specific holdout, calibration, SHAP and subgroup reports.

## Run locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` or call:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @docs/api/sample_request.json
```

Prediction 1 example:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @docs/api/sample_high_risk_request.json
```

Response:

```json
{
  "model_version": "huy-catboost-1.0.0",
  "risk_score": 0.9379868301418867,
  "decision_threshold": 0.8564852152742759,
  "prediction": 1,
  "status": "high_risk"
}
```

Object key order in JSON does not affect inference. Field names, types and values must satisfy the
schema. The backend performs medication encoding, diagnosis grouping, log transforms, interactions
and scaling; clients must not submit the 52 engineered features.

## Verification

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python scripts/smoke_test.py
```

## Docker Compose

```bash
docker compose build
docker compose up -d
docker compose ps
```

Services: API `8000`, MLflow `5050`, Prometheus `9090`, Grafana `3000`.

## Reproduced final test metrics

| Metric | Value |
| --- | ---: |
| PR-AUC | 0.1081018905 |
| ROC-AUC | 0.5667913495 |
| Precision | 0.1585760518 |
| Recall | 0.0512552301 |
| F1 | 0.0774703557 |
| Brier score | 0.3855983188 |

Confusion matrix: TN 10,115; FP 260; FN 907; TP 49.

These results reproduce the saved notebook output. The low recall, lack of calibration, preprocessing
leakage and resampling protocol are documented limitations; this system must not be used for
autonomous clinical decisions.
