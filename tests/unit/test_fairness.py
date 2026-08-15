import numpy as np
import pandas as pd
import pytest

from src.evaluation.fairness import (
    _safe_ranking_metric,
    subgroup_metrics,
    summarize_fairness_gaps,
)


def test_safe_ranking_metric_returns_metric_with_both_classes() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = _safe_ranking_metric(
        lambda y, s: 0.75,
        y_true,
        scores,
    )

    assert result == 0.75


def test_safe_ranking_metric_returns_nan_for_single_class() -> None:
    y_true = np.array([0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3])

    result = _safe_ranking_metric(
        lambda y, s: 0.75,
        y_true,
        scores,
    )

    assert np.isnan(result)


def test_subgroup_metrics_calculates_expected_values() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.10, 0.20, 0.90, 0.60])
    predictions = np.array([0, 0, 1, 0])

    result = subgroup_metrics(
        attribute="gender",
        group="Female",
        y_true=y_true,
        scores=scores,
        predictions=predictions,
    )

    assert result["attribute"] == "gender"
    assert result["group"] == "Female"

    assert result["n"] == 4
    assert result["positives"] == 2
    assert result["prevalence"] == 0.5

    assert result["precision"] == 1.0
    assert result["recall_tpr"] == 0.5
    assert result["fpr"] == 0.0
    assert result["specificity"] == 1.0

    assert result["predicted_positive_rate"] == 0.25

    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 1
    assert result["tn"] == 2

    assert result["small_sample_caution"] is True


def test_subgroup_metrics_handles_zero_denominators() -> None:
    # No actual positives and no predicted positives.
    y_true = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    predictions = np.array([0, 0, 0, 0])

    result = subgroup_metrics(
        attribute="race",
        group="Asian",
        y_true=y_true,
        scores=scores,
        predictions=predictions,
    )

    assert result["positives"] == 0
    assert result["precision"] == 0.0
    assert result["recall_tpr"] == 0.0
    assert result["fpr"] == 0.0
    assert result["specificity"] == 1.0
    assert result["predicted_positive_rate"] == 0.0

    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 4


def test_summarize_fairness_gaps_uses_only_reliable_groups() -> None:
    report = pd.DataFrame(
        [
            {
                "attribute": "gender",
                "group": "Female",
                "small_sample_caution": False,
                "recall_tpr": 0.80,
                "fpr": 0.10,
                "predicted_positive_rate": 0.30,
            },
            {
                "attribute": "gender",
                "group": "Male",
                "small_sample_caution": False,
                "recall_tpr": 0.60,
                "fpr": 0.20,
                "predicted_positive_rate": 0.50,
            },
            {
                "attribute": "gender",
                "group": "Missing",
                "small_sample_caution": True,
                "recall_tpr": 0.10,
                "fpr": 0.90,
                "predicted_positive_rate": 0.90,
            },
        ]
    )

    result = summarize_fairness_gaps(report)

    assert len(result) == 1

    row = result.iloc[0]

    assert row["attribute"] == "gender"
    assert row["reliable_group_count"] == 2
    assert row["reliable_groups"] == "Female|Male"

    assert row["recall_tpr_min"] == pytest.approx(0.60)
    assert row["recall_tpr_max"] == pytest.approx(0.80)
    assert row["recall_tpr_gap"] == pytest.approx(0.20)

    assert row["fpr_min"] == pytest.approx(0.10)
    assert row["fpr_max"] == pytest.approx(0.20)
    assert row["fpr_gap"] == pytest.approx(0.10)

    assert row["predicted_positive_rate_min"] == pytest.approx(0.30)
    assert row["predicted_positive_rate_max"] == pytest.approx(0.50)
    assert row["predicted_positive_rate_gap"] == pytest.approx(0.20)
