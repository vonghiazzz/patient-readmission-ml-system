"""Native CatBoost SHAP audit for Huy's final model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

from src.api.dependencies import load_production_artifacts
from src.evaluation.data_contract import align_source_features
from src.features.build_features import CATEGORICAL_MODEL_FEATURES


def run_global_shap(
    input_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    sample_size: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = load_production_artifacts(artifact_dir)
    evaluation = pd.read_csv(input_path, low_memory=False)
    sample = evaluation.sample(n=min(sample_size, len(evaluation)), random_state=random_state)
    model_input = align_source_features(sample, artifacts.preprocessing_state)
    pool = Pool(model_input, cat_features=list(CATEGORICAL_MODEL_FEATURES))
    shap_with_base = np.asarray(
        artifacts.model.get_feature_importance(pool, type="ShapValues"), dtype=float
    )
    shap_values = shap_with_base[:, :-1]
    if shap_values.shape != model_input.shape:
        raise ValueError(f"Unexpected CatBoost SHAP shape: {shap_values.shape}")

    importance = pd.DataFrame(
        {
            "model_feature": model_input.columns,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_dir / "global_shap_huy_features.csv", index=False)

    top = importance.head(20).sort_values("mean_abs_shap")
    top.plot.barh(x="model_feature", y="mean_abs_shap", legend=False, figsize=(9, 7))
    plt.xlabel("Mean absolute CatBoost SHAP value")
    plt.title("Huy final CatBoost — global feature importance")
    plt.tight_layout()
    plt.savefig(output_dir / "global_shap_huy_features.png", dpi=180)
    plt.close()
    return importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/production_huy"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/production_huy/reports"))
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
