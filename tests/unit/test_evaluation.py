import json
from pathlib import Path

import numpy as np

from src.evaluation.fairness import subgroup_metrics


def test_single_class_fairness_slice_is_safe_and_flagged() -> None:
    result = subgroup_metrics(
        attribute="race",
        group="small",
        y_true=np.zeros(10, dtype=int),
        scores=np.linspace(0.01, 0.10, 10),
        predictions=np.zeros(10, dtype=int),
    )
    assert np.isnan(result["pr_auc"])
    assert np.isnan(result["roc_auc"])
    assert result["small_sample_caution"] is True
    assert result["tn"] == 10


def test_huy_holdout_report_matches_notebook_confusion_matrix() -> None:
    report = json.loads(
        Path("models/production_huy/reports/evaluation_metrics.json").read_text(encoding="utf-8")
    )
    assert report["test_rows"] == 11331
    assert report["confusion_matrix"] == {"tn": 10115, "fp": 260, "fn": 907, "tp": 49}
    assert report["decision_threshold"] == 0.8564852152742759


def test_huy_specific_evaluation_reports_exist() -> None:
    reports = Path("models/production_huy/reports")
    expected = {
        "evaluation_metrics.json",
        "calibration_curve.csv",
        "global_shap_huy_features.csv",
        "global_shap_huy_features.png",
        "subgroup_fairness_report.csv",
        "fairness_gap_summary.csv",
    }
    assert expected.issubset({path.name for path in reports.iterdir()})
