# Development Workflow

Branch flow: `feature/*` → pull request → `develop` → release pull request → `main`.

## Local quality gate

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

## Huy data/evaluation reproduction

Raw data remains local at `data/raw/diabetic_data.csv`.

```bash
python -m src.data.splitting
python -m src.models.evaluate
python -m src.evaluation.explainability --input data/interim/splits/test.csv
python -m src.evaluation.fairness --input data/interim/splits/test.csv
```

Do not fit artifacts during API startup. Changes to model, preprocessing state, feature manifest or
threshold require a new model version, reference predictions and production-contract tests.

## Service smoke tests

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
python scripts/smoke_test.py
docker compose config
docker compose build
```

Never commit raw/interim patient data, `.env`, caches or local MLflow storage.
