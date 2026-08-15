import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.data_contract import align_source_features
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


def test_align_source_features_resolves_normalized_names() -> None:
    sample = json.loads(Path("docs/api/sample_request.json").read_text(encoding="utf-8"))

    frame = pd.DataFrame([sample])

    # Simulate source columns using hyphens instead of underscores.
    renamed = {
        "admission_type_id": "admission-type-id",
        "discharge_disposition_id": "discharge-disposition-id",
        "admission_source_id": "admission-source-id",
        "time_in_hospital": "time-in-hospital",
        "num_lab_procedures": "num-lab-procedures",
        "num_procedures": "num-procedures",
        "num_medications": "num-medications",
        "number_outpatient": "number-outpatient",
        "number_emergency": "number-emergency",
        "number_inpatient": "number-inpatient",
        "number_diagnoses": "number-diagnoses",
        "diag_1": "diag-1",
    }

    frame = frame.rename(columns=renamed)

    preprocessing_state = json.loads(
        Path("models/production_huy/preprocessing_state.json").read_text(encoding="utf-8")
    )

    result = align_source_features(frame, preprocessing_state)

    assert result.shape[0] == 1
    assert not result.empty


def test_align_source_features_with_sample_request() -> None:
    sample = json.loads(Path("docs/api/sample_request.json").read_text(encoding="utf-8"))

    frame = pd.DataFrame([sample])

    preprocessing_state = json.loads(
        Path("models/production_huy/preprocessing_state.json").read_text(encoding="utf-8")
    )

    result = align_source_features(frame, preprocessing_state)

    assert result.shape[0] == 1
    assert result.shape[1] > 0


def test_align_source_features_fails_when_feature_missing() -> None:
    sample = json.loads(Path("docs/api/sample_request.json").read_text(encoding="utf-8"))

    sample.pop("race")

    frame = pd.DataFrame([sample])

    preprocessing_state = json.loads(
        Path("models/production_huy/preprocessing_state.json").read_text(encoding="utf-8")
    )

    with pytest.raises(ValueError, match="cannot resolve Huy raw features"):
        align_source_features(frame, preprocessing_state)
