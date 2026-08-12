"""Descriptive subgroup audit for Huy's model and notebook threshold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from src.api.dependencies import load_production_artifacts
from src.evaluation.data_contract import align_source_features

AUDIT_ATTRIBUTES = ("race", "gender", "age")


def _safe_ranking_metric(metric: Any, y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(metric(y_true, scores))


def subgroup_metrics(
    attribute: str,
    group: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    predicted_positives = int(predictions.sum())
    return {
        "attribute": attribute,
        "group": group,
        "n": int(len(y_true)),
        "positives": positives,
        "prevalence": float(y_true.mean()),
        "pr_auc": _safe_ranking_metric(average_precision_score, y_true, scores),
        "roc_auc": _safe_ranking_metric(roc_auc_score, y_true, scores),
        "brier": float(brier_score_loss(y_true, scores)),
        "precision": float(tp / predicted_positives) if predicted_positives else 0.0,
        "recall_tpr": float(tp / positives) if positives else 0.0,
        "fpr": float(fp / negatives) if negatives else 0.0,
        "specificity": float(tn / negatives) if negatives else 0.0,
        "predicted_positive_rate": float(predictions.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "small_sample_caution": bool(len(y_true) < 200 or positives < 30),
    }


def summarize_fairness_gaps(report: pd.DataFrame) -> pd.DataFrame:
    """Summarize ranges for reliable groups; this is not a fairness verdict."""

    rows: list[dict[str, Any]] = []
    reliable = report.loc[~report["small_sample_caution"].astype(bool)]
    for attribute, groups in reliable.groupby("attribute", sort=False):
        row: dict[str, Any] = {
            "attribute": attribute,
            "reliable_group_count": int(len(groups)),
            "reliable_groups": "|".join(groups["group"].astype(str)),
        }
        for metric in ("recall_tpr", "fpr", "predicted_positive_rate"):
            row[f"{metric}_min"] = float(groups[metric].min())
            row[f"{metric}_max"] = float(groups[metric].max())
            row[f"{metric}_gap"] = float(groups[metric].max() - groups[metric].min())
        rows.append(row)
    return pd.DataFrame(rows)


def run_fairness_audit(
    input_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    target_column: str = "readmitted_30d",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = load_production_artifacts(artifact_dir)
    evaluation = pd.read_csv(input_path, low_memory=False)
    if target_column not in evaluation:
        raise ValueError(f"Fairness input is missing target column: {target_column}")

    model_input = align_source_features(evaluation, artifacts.preprocessing_state)
    scores = np.asarray(artifacts.model.predict_proba(model_input), dtype=float)[:, 1]
    predictions = (scores >= artifacts.decision_threshold).astype(int)
    y_true = evaluation[target_column].astype(int).to_numpy()

    rows: list[dict[str, Any]] = []
    for attribute in AUDIT_ATTRIBUTES:
        groups = evaluation[attribute].fillna("Missing").astype(str)
        for group in sorted(groups.unique()):
            mask = groups.eq(group).to_numpy()
            rows.append(
                subgroup_metrics(attribute, group, y_true[mask], scores[mask], predictions[mask])
            )

    report = pd.DataFrame(rows)
    gaps = summarize_fairness_gaps(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "subgroup_fairness_report.csv", index=False)
    gaps.to_csv(output_dir / "fairness_gap_summary.csv", index=False)
    return report, gaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/production_huy"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/production_huy/reports"))
    parser.add_argument("--target-column", default="readmitted_30d")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_fairness_audit(
        input_path=arguments.input,
        artifact_dir=arguments.artifact_dir,
        output_dir=arguments.output_dir,
        target_column=arguments.target_column,
    )
