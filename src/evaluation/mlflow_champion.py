"""Record the already-frozen champion in MLflow without fitting any model."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.api.dependencies import load_production_artifacts

RUN_NAME = "xgb-v1-production-1.0.0"


def _existing_run_id(client: MlflowClient, experiment_id: str) -> str | None:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=("tags.model_version = '1.0.0' AND tags.selection_status = 'champion'"),
    )
    if len(runs) > 1:
        raise RuntimeError("Multiple final champion MLflow runs already exist")
    return runs[0].info.run_id if runs else None


def log_frozen_champion(
    artifact_dir: Path,
    tracking_uri: str,
    experiment_name: str,
) -> str:
    """Log metadata, metrics, and copies of frozen artifacts; never retrain."""

    artifacts = load_production_artifacts(artifact_dir)
    metadata = artifacts.metadata
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    existing_run_id = _existing_run_id(client, experiment.experiment_id)

    with mlflow.start_run(run_id=existing_run_id, run_name=RUN_NAME) as run:
        mlflow.set_tags(
            {
                "stage": "production_candidate",
                "feature_set": str(metadata["feature_set"]),
                "selection_status": "champion",
                "model_version": str(metadata["model_version"]),
            }
        )
        mlflow.log_params(
            {
                "best_iteration": int(metadata["best_iteration"]),
                "decision_threshold": float(metadata["decision_threshold"]),
                "calibration": str(metadata["posthoc_calibration"]),
            }
        )
        # These values are read from the serialized champion, not reconstructed
        # or guessed from the study history.
        model_parameters = artifacts.model.get_params()
        trial_parameter_names = (
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "scale_pos_weight",
        )
        mlflow.log_params(
            {
                f"trial22_{name}": model_parameters[name]
                for name in trial_parameter_names
                if model_parameters.get(name) is not None
            }
        )
        for name, value in metadata["validation_metrics"].items():
            mlflow.log_metric(f"validation_{name}", float(value))
        for name, value in metadata["final_holdout_metrics"].items():
            mlflow.log_metric(f"final_{name}", float(value))

        for filename in (
            "model.joblib",
            "preprocessor.joblib",
            "metadata.json",
            "feature_manifest.json",
        ):
            mlflow.log_artifact(str(artifacts.artifact_dir / filename), artifact_path="artifacts")

        reports_dir = artifacts.artifact_dir / "reports"
        for filename in (
            "global_shap_original_features.csv",
            "shap_beeswarm_transformed.png",
            "subgroup_fairness_report.csv",
            "fairness_gap_summary.csv",
        ):
            report = reports_dir / filename
            if report.is_file():
                mlflow.log_artifact(str(report), artifact_path="reports")
        return run.info.run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/production_v1"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns/mlflow.db")
    parser.add_argument("--experiment-name", default="patient-readmission-production")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_id = log_frozen_champion(
        artifact_dir=arguments.artifact_dir,
        tracking_uri=arguments.tracking_uri,
        experiment_name=arguments.experiment_name,
    )
    print(f"Frozen champion logged to MLflow run: {run_id}")
