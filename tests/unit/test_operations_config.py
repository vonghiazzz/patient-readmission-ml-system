import json
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_prometheus_scrape_and_alert_rules_are_configured() -> None:
    prometheus = load_yaml("monitoring/prometheus.yml")
    scrape = prometheus["scrape_configs"][0]
    assert scrape["job_name"] == "readmission-api"
    assert scrape["metrics_path"] == "/metrics"
    assert scrape["static_configs"][0]["targets"] == ["api:8000"]

    rules = load_yaml("monitoring/alert_rules.yml")["groups"][0]["rules"]
    assert {rule["alert"] for rule in rules} == {
        "ReadmissionApiUnavailable",
        "ReadmissionModelNotReady",
        "ReadmissionApiElevatedErrorRate",
        "ReadmissionApiHighLatency",
    }


def test_grafana_datasource_and_required_dashboard_panels_are_provisioned() -> None:
    datasource = load_yaml("monitoring/grafana/provisioning/datasources/prometheus.yml")[
        "datasources"
    ][0]
    assert datasource["uid"] == "prometheus"
    assert datasource["url"] == "http://prometheus:9090"

    dashboard = json.loads(
        Path("monitoring/grafana/dashboards/patient-readmission.json").read_text(encoding="utf-8")
    )
    panel_titles = {panel["title"] for panel in dashboard["panels"]}
    assert {
        "Request volume",
        "Request latency",
        "Error rate",
        "Prediction-risk distribution",
    }.issubset(panel_titles)


def test_dockerfile_packages_real_bundle_and_runs_non_root() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    for filename in (
        "model.joblib",
        "preprocessor.joblib",
        "feature_manifest.json",
        "metadata.json",
        "cat_tunning_model.pkl",
    ):
        assert f"models/production_v1/{filename}" in dockerfile
    assert "USER appuser" in dockerfile
    assert "models/unavailable" not in dockerfile
    assert "src.models.train" not in dockerfile
