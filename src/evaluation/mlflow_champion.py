"""Log the already-frozen Huy CatBoost champion without fitting a model."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.api.dependencies import load_production_artifacts

RUN_NAME = "huy-catboost-production-1.0.0"


def _existing_run_id(client: MlflowClient, experiment_id: str, model_version: str) -> str | None:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.model_version = '{model_version}' AND tags.selection_status = 'champion'"
        ),
    )
    if len(runs) > 1:
        raise RuntimeError("Multiple Huy champion MLflow runs already exist")
    return runs[0].info.run_id if runs else None


def log_frozen_champion(
    artifact_dir: Path,
    tracking_uri: str,
    experiment_name: str,
) -> str:
    artifacts = load_production_artifacts(artifact_dir)
    metadata = artifacts.metadata
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    existing_run_id = _existing_run_id(client, experiment.experiment_id, artifacts.model_version)

    with mlflow.start_run(run_id=existing_run_id, run_name=RUN_NAME) as run:
        mlflow.set_tags(
            {
                "stage": "production_candidate",
                "feature_set": str(metadata["feature_set"]),
                "selection_status": "champion",
                "model_version": artifacts.model_version,
                "model_type": str(metadata["model_type"]),
                "source_notebook": str(metadata["source_notebook"]),
            }
        )
        mlflow.log_params(
            {
                **metadata["hyperparameters"],
                "decision_threshold": float(metadata["decision_threshold"]),
                "calibration": str(metadata["posthoc_calibration"]),
                "model_sha256": artifacts.model_sha256,
            }
        )
        for name, value in metadata["final_test_metrics"].items():
            mlflow.log_metric(f"final_{name}", float(value))

        for filename in (
            "model.pkl",
            "preprocessing_state.json",
            "metadata.json",
            "feature_manifest.json",
            "reference_predictions.json",
        ):
            mlflow.log_artifact(str(artifacts.artifact_dir / filename), artifact_path="artifacts")
        reports_dir = artifacts.artifact_dir / "reports"
        if reports_dir.is_dir():
            mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
        return run.info.run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/production_huy"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns/mlflow.db")
    parser.add_argument("--experiment-name", default="patient-readmission-huy-production")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_id = log_frozen_champion(
        artifact_dir=arguments.artifact_dir,
        tracking_uri=arguments.tracking_uri,
        experiment_name=arguments.experiment_name,
    )
    print(f"Huy champion logged to MLflow run: {run_id}")
