import numpy as np

from src.evaluation.explainability import compute_shap_values_in_batches, group_shap_per_sample
from src.evaluation.fairness import subgroup_metrics


def test_shap_groups_per_sample_before_absolute_mean() -> None:
    # The two OHE columns cancel per patient. Summing mean absolute values per
    # transformed column would incorrectly report a non-zero importance.
    values = np.array([[2.0, -2.0, 1.0], [-3.0, 3.0, -1.0]])
    grouped = group_shap_per_sample(
        values,
        raw_features=["categorical", "numeric"],
        transformed_mapping={"categorical": [0, 1], "numeric": [2]},
    )
    np.testing.assert_array_equal(grouped, np.array([[0.0, 1.0], [0.0, -1.0]]))
    np.testing.assert_array_equal(np.mean(np.abs(grouped), axis=0), np.array([0.0, 1.0]))


def test_shap_batching_preserves_row_order() -> None:
    class Explanation:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

    class Explainer:
        def __call__(self, values: np.ndarray, check_additivity: bool) -> Explanation:
            assert check_additivity is False
            return Explanation(values * 2)

    transformed = np.arange(21, dtype=float).reshape(7, 3)
    result = compute_shap_values_in_batches(Explainer(), transformed, batch_size=2)
    np.testing.assert_array_equal(result, transformed * 2)


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
