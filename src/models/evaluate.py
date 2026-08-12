"""Reproduce Huy's final holdout evaluation from raw data and frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.api.dependencies import load_production_artifacts
from src.data.splitting import create_huy_cohort, create_huy_splits
from src.evaluation.data_contract import align_source_features


def evaluate_huy_holdout(
    raw_data_path: Path,
    artifact_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    artifacts = load_production_artifacts(artifact_dir)
    raw = pd.read_csv(raw_data_path, low_memory=False)
    cohort = create_huy_cohort(raw, dict(artifacts.preprocessing_state))
    _, test = create_huy_splits(cohort)
    model_input = align_source_features(test, artifacts.preprocessing_state)
    target = test["readmitted_30d"].to_numpy(dtype=int)
    scores = np.asarray(artifacts.model.predict_proba(model_input), dtype=float)[:, 1]
    predictions = (scores >= artifacts.decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()

    metrics: dict[str, Any] = {
        "model_version": artifacts.model_version,
        "model_sha256": artifacts.model_sha256,
        "decision_threshold": artifacts.decision_threshold,
        "test_rows": int(len(test)),
        "negative_count": int((target == 0).sum()),
        "positive_count": int((target == 1).sum()),
        "precision": float(precision_score(target, predictions)),
        "recall": float(recall_score(target, predictions)),
        "f1": float(f1_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "pr_auc": float(average_precision_score(target, scores)),
        "brier_score": float(brier_score_loss(target, scores)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    fraction_positive, mean_predicted = calibration_curve(
        target, scores, n_bins=10, strategy="quantile"
    )
    calibration = pd.DataFrame(
        {"mean_predicted_probability": mean_predicted, "fraction_positive": fraction_positive}
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    calibration.to_csv(output_dir / "calibration_curve.csv", index=False)
    return metrics, calibration, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/diabetic_data.csv"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/production_huy"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/production_huy/reports"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result, _, _ = evaluate_huy_holdout(
        arguments.input, arguments.artifact_dir, arguments.output_dir
    )
    print(json.dumps(result, indent=2))
