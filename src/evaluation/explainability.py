"""Reproducible global SHAP analysis for the frozen XGBoost champion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api.dependencies import ProductionArtifacts, load_production_artifacts
from src.evaluation.data_contract import align_source_features


def transformed_columns_by_raw_feature(preprocessor: Any) -> dict[str, list[int]]:
    """Map each raw input to its transformed output indices."""

    mapping: dict[str, list[int]] = {}
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder" or isinstance(columns, str):
            continue
        columns = list(columns)
        output_slice = preprocessor.output_indices_[transformer_name]
        start = int(output_slice.start)

        if transformer_name == "numeric":
            for offset, column in enumerate(columns):
                mapping[column] = [start + offset]
            continue

        encoder = transformer.named_steps.get("encoder")
        if encoder is None:
            raise ValueError("Categorical pipeline has no fitted encoder")
        cursor = start
        drop_indices = getattr(encoder, "drop_idx_", None)
        for index, (column, categories) in enumerate(
            zip(columns, encoder.categories_, strict=True)
        ):
            width = len(categories)
            if drop_indices is not None and drop_indices[index] is not None:
                width -= 1
            mapping[column] = list(range(cursor, cursor + width))
            cursor += width
        if cursor != int(output_slice.stop):
            raise ValueError("Categorical transformed-column mapping is inconsistent")
    return mapping


def group_shap_per_sample(
    shap_values: np.ndarray,
    raw_features: list[str],
    transformed_mapping: dict[str, list[int]],
) -> np.ndarray:
    """Sum transformed SHAP values per raw feature before abs/mean aggregation."""

    grouped = np.empty((shap_values.shape[0], len(raw_features)), dtype=float)
    for feature_index, feature in enumerate(raw_features):
        indices = transformed_mapping[feature]
        grouped[:, feature_index] = shap_values[:, indices].sum(axis=1)
    return grouped


def compute_shap_values_in_batches(
    explainer: Any,
    transformed: np.ndarray,
    batch_size: int = 250,
) -> np.ndarray:
    """Bound native SHAP memory while retaining one 2-D result matrix."""

    batches: list[np.ndarray] = []
    for start in range(0, transformed.shape[0], batch_size):
        stop = min(start + batch_size, transformed.shape[0])
        explanation = explainer(transformed[start:stop], check_additivity=False)
        batches.append(np.asarray(explanation.values))
    return np.concatenate(batches, axis=0)


def run_global_shap(
    input_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    sample_size: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate transformed beeswarm and correctly grouped raw importance."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    artifacts: ProductionArtifacts = load_production_artifacts(artifact_dir)
    evaluation = pd.read_csv(input_path, low_memory=False)
    if len(evaluation) < sample_size:
        raise ValueError(f"SHAP input requires at least {sample_size} rows")
    sample = evaluation.sample(n=sample_size, random_state=random_state)
    model_input = align_source_features(sample, artifacts.feature_manifest)
    transformed = artifacts.preprocessor.transform(model_input)
    if transformed.shape != (sample_size, artifacts.transformed_feature_count):
        raise ValueError("Unexpected transformed SHAP sample shape")

    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    explainer = shap.TreeExplainer(artifacts.model)
    values = compute_shap_values_in_batches(explainer, dense)
    if values.shape != dense.shape:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    transformed_names = artifacts.preprocessor.get_feature_names_out().tolist()
    explanation = shap.Explanation(values=values, data=dense, feature_names=transformed_names)
    shap.plots.beeswarm(explanation, max_display=25, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm_transformed.png", dpi=180, bbox_inches="tight")
    plt.close()

    raw_features = list(artifacts.feature_manifest["model_input_features"])
    mapping = transformed_columns_by_raw_feature(artifacts.preprocessor)
    grouped = group_shap_per_sample(values, raw_features, mapping)
    importance = pd.DataFrame(
        {
            "raw_feature": raw_features,
            "mean_abs_grouped_shap": np.mean(np.abs(grouped), axis=0),
        }
    ).sort_values("mean_abs_grouped_shap", ascending=False)
    importance.to_csv(output_dir / "global_shap_original_features.csv", index=False)

    top = importance.head(20).sort_values("mean_abs_grouped_shap")
    top.plot.barh(x="raw_feature", y="mean_abs_grouped_shap", legend=False, figsize=(9, 7))
    plt.xlabel("Mean absolute grouped SHAP value")
    plt.tight_layout()
    plt.savefig(output_dir / "global_shap_original_features.png", dpi=180)
    plt.close()
    return importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/production_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/production_v1/reports"))
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_global_shap(
        input_path=arguments.input,
        artifact_dir=arguments.artifact_dir,
        output_dir=arguments.output_dir,
        sample_size=arguments.sample_size,
        random_state=arguments.random_state,
    )
