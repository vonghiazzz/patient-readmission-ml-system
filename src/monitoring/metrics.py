"""Low-cardinality Prometheus instrumentation for the online API."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

HTTP_REQUESTS = Counter(
    "readmission_http_requests_total",
    "HTTP requests handled by the API.",
    ("method", "route", "status_code"),
)
HTTP_ERRORS = Counter(
    "readmission_http_errors_total",
    "HTTP responses with status code 400 or greater.",
    ("method", "route", "status_code"),
)
HTTP_LATENCY = Histogram(
    "readmission_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "route"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
PREDICTIONS = Counter(
    "readmission_predictions_total",
    "Successful predictions by bounded model identifier.",
    ("model", "prediction"),
)
PREDICTION_RISK_SCORE = Histogram(
    "readmission_prediction_risk_score",
    "Distribution of model predict_proba scores.",
    ("model",),
    buckets=(0.02, 0.05, 0.1, 0.17, 0.25, 0.4, 0.6, 0.8, 1.0),
)
MODEL_READY = Gauge(
    "readmission_model_ready",
    "Whether the frozen production artifact contract is ready (1) or unavailable (0).",
)
MODEL_INFO = Gauge(
    "readmission_model_info",
    "Frozen model information; value is always one for the loaded bundle.",
    ("model_version", "feature_set"),
)


def configure_model_info(
    ready: bool,
    model_version: str | None = None,
    feature_set: str | None = None,
) -> None:
    MODEL_READY.set(1 if ready else 0)
    MODEL_INFO.clear()
    if ready and model_version and feature_set:
        MODEL_INFO.labels(model_version=model_version, feature_set=feature_set).set(1)


def set_model_readiness(
    ready: bool,
    model_version: str | None = None,
    feature_set: str | None = None,
) -> None:
    configure_model_info(ready, model_version, feature_set)


def record_prediction(risk_score: float, prediction: int, model: str) -> None:
    PREDICTIONS.labels(model=model, prediction=str(prediction)).inc()
    PREDICTION_RISK_SCORE.labels(model=model).observe(risk_score)


def _route_label(request: Request) -> str:
    route: Any = request.scope.get("route")
    return str(getattr(route, "path", "unmatched"))


async def http_metrics_middleware(request: Request, call_next: Any) -> Response:
    start = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = _route_label(request)
        labels = {
            "method": request.method,
            "route": route,
            "status_code": str(status_code),
        }
        HTTP_REQUESTS.labels(**labels).inc()
        if status_code >= 400:
            HTTP_ERRORS.labels(**labels).inc()
        HTTP_LATENCY.labels(method=request.method, route=route).observe(perf_counter() - start)


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
