import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.api.dependencies import load_production_artifacts
from src.api.schemas import PredictionRequest
from src.config.settings import get_settings
from src.features.build_features import DERIVED_FEATURES, build_v1_features


def artifacts():
    return load_production_artifacts(get_settings().production_artifact_dir)


def valid_payload() -> dict[str, Any]:
    bundle = artifacts()
    payload: dict[str, Any] = {}
    for name, transformer, columns in bundle.preprocessor.transformers_:
        if name == "remainder" or isinstance(columns, str):
            continue
        if name == "numeric":
            for column in columns:
                if column in bundle.feature_manifest["request_features"]:
                    payload[column] = 1
        else:
            encoder = transformer.named_steps["encoder"]
            for column, categories in zip(columns, encoder.categories_, strict=True):
                if column not in bundle.feature_manifest["request_features"]:
                    continue
                value = next(value for value in categories.tolist() if str(value) != "nan")
                payload[column] = value.item() if hasattr(value, "item") else value
    return {name: payload[name] for name in bundle.feature_manifest["request_features"]}


def test_feature_manifest_counts_and_composition() -> None:
    manifest = artifacts().feature_manifest
    assert len(manifest["request_features"]) == 42
    assert tuple(manifest["derived_features"]) == DERIVED_FEATURES
    assert len(manifest["model_input_features"]) == 45
    assert manifest["model_input_features"] == (
        manifest["request_features"] + manifest["derived_features"]
    )


def test_feature_builder_adds_only_three_binary_history_flags() -> None:
    frame = pd.DataFrame(
        {
            "number_outpatient": [0, 2],
            "number_emergency": [0, 3],
            "number_inpatient": [0, 4],
        }
    )
    result = build_v1_features(frame)
    assert result.columns.tolist() == frame.columns.tolist() + list(DERIVED_FEATURES)
    assert result[list(DERIVED_FEATURES)].values.tolist() == [[0, 0, 0], [1, 1, 1]]
    assert result[list(DERIVED_FEATURES)].dtypes.apply(str).eq("int8").all()


def test_schema_and_input_order_match_manifest() -> None:
    bundle = artifacts()
    assert list(PredictionRequest.model_fields) == bundle.feature_manifest["request_features"]
    model_input = bundle.prepare_model_input(valid_payload())
    assert model_input.columns.tolist() == bundle.feature_manifest["model_input_features"]


def test_all_four_artifacts_reload_and_dimensions_match() -> None:
    bundle = artifacts()
    assert joblib.load(bundle.artifact_dir / "model.joblib").n_features_in_ == 223
    assert joblib.load(bundle.artifact_dir / "preprocessor.joblib").n_features_in_ == 45
    assert bundle.transformed_feature_count == 223
    assert bundle.model_version == "1.0.0"


def test_reloaded_artifacts_reproduce_prediction() -> None:
    bundle = artifacts()
    first = bundle.predict(valid_payload())
    np.testing.assert_allclose(first.risk_score, 0.1169060543179512, rtol=0, atol=1e-9)
    load_production_artifacts.cache_clear()
    reloaded = artifacts().predict(valid_payload())
    np.testing.assert_allclose(reloaded.risk_score, first.risk_score, rtol=0, atol=1e-12)
    assert reloaded.prediction == first.prediction


def test_unknown_category_and_permitted_missing_category_do_not_crash() -> None:
    bundle = artifacts()
    unknown = valid_payload()
    unknown["race"] = "NeverSeenRace"
    assert np.isfinite(bundle.predict(unknown).risk_score)

    missing = valid_payload()
    missing["race"] = None
    validated = PredictionRequest.model_validate(missing)
    assert np.isfinite(bundle.predict(validated.model_dump()).risk_score)


def test_frozen_files_are_present_at_single_source_path() -> None:
    root = Path("models/production_v1")
    assert {path.name for path in root.glob("*.joblib")} == {
        "model.joblib",
        "preprocessor.joblib",
    }
    assert {path.name for path in root.glob("*.json")} == {
        "feature_manifest.json",
        "metadata.json",
    }


def test_documented_synthetic_request_matches_real_contract() -> None:
    payload = json.loads(Path("docs/api/sample_request.json").read_text(encoding="utf-8"))
    validated = PredictionRequest.model_validate(payload)
    assert list(validated.model_dump()) == artifacts().feature_manifest["request_features"]
    result = artifacts().predict(validated.model_dump())
    assert 0 <= result.risk_score <= 1
    assert result.prediction == int(result.risk_score >= artifacts().decision_threshold)
