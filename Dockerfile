FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser models/production_v1/model.joblib ./models/production_v1/model.joblib
COPY --chown=appuser:appuser models/production_v1/preprocessor.joblib ./models/production_v1/preprocessor.joblib
COPY --chown=appuser:appuser models/production_v1/feature_manifest.json ./models/production_v1/feature_manifest.json
COPY --chown=appuser:appuser models/production_v1/metadata.json ./models/production_v1/metadata.json
COPY --chown=appuser:appuser models/production_v1/cat_tunning_model.pkl ./models/production_v1/cat_tunning_model.pkl
COPY --chown=appuser:appuser models/production_v1/reports ./models/production_v1/reports

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
