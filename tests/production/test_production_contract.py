import json
from pathlib import Path

import numpy as np

from src.api.dependencies import load_production_artifacts
from src.api.schemas import PredictionRequest
from src.config.settings import get_settings
from src.features.build_features import (
    CATEGORICAL_MODEL_FEATURES,
    MODEL_INPUT_FEATURES,
    REQUEST_FEATURES,
)


def artifacts():
    return load_production_artifacts(get_settings().production_artifact_dir)


def payload(filename: str = "sample_request.json") -> dict:
    value = json.loads(Path("docs/api", filename).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_huy_manifest_and_embedded_model_contract_match() -> None:
    bundle = artifacts()
    assert tuple(bundle.feature_manifest["request_features"]) == REQUEST_FEATURES
    assert tuple(bundle.feature_manifest["model_input_features"]) == MODEL_INPUT_FEATURES
    assert tuple(bundle.feature_manifest["categorical_model_features"]) == (
        CATEGORICAL_MODEL_FEATURES
    )
    assert len(REQUEST_FEATURES) == 40
    assert len(MODEL_INPUT_FEATURES) == 52
    assert tuple(bundle.model.feature_names_) == MODEL_INPUT_FEATURES
    categorical_indices = bundle.model.get_cat_feature_indices()
    assert tuple(MODEL_INPUT_FEATURES[index] for index in categorical_indices) == (
        CATEGORICAL_MODEL_FEATURES
    )


def test_final_model_identity_and_threshold_are_frozen() -> None:
    bundle = artifacts()
    assert bundle.model_version == "huy-catboost-1.0.0"
    assert bundle.decision_threshold == 0.8564852152742759
    assert bundle.model_sha256 == (
        "a0d11b7ed0c1956d10afbfda360ec24ae2c55f6d6d50d32ed50780a81160331b"
    )
    assert bundle.metadata["model_sha256"] == bundle.model_sha256
    assert bundle.model.get_params()["class_weights"] == {0: 1, 1: 10}


def test_raw_request_is_transformed_to_finite_52_feature_frame() -> None:
    bundle = artifacts()
    validated = PredictionRequest.model_validate(payload())
    frame = bundle.prepare_model_input(validated.model_dump())
    assert tuple(frame.columns) == MODEL_INPUT_FEATURES
    assert frame.shape == (1, 52)
    numeric = frame.drop(columns=list(CATEGORICAL_MODEL_FEATURES)).to_numpy(dtype=float)
    assert np.isfinite(numeric).all()


def test_reference_predictions_zero_and_one_reproduce_after_reload() -> None:
    bundle = artifacts()
    low = bundle.predict(payload())
    high = bundle.predict(payload("sample_high_risk_request.json"))
    np.testing.assert_allclose(low.risk_score, 0.6643320154788986, rtol=0, atol=1e-12)
    assert low.prediction == 0
    np.testing.assert_allclose(high.risk_score, 0.9379868301418867, rtol=0, atol=1e-12)
    assert high.prediction == 1

    load_production_artifacts.cache_clear()
    reloaded = artifacts().predict(payload())
    np.testing.assert_allclose(reloaded.risk_score, low.risk_score, rtol=0, atol=1e-12)


def test_only_huy_notebook_and_bundle_are_authoritative() -> None:
    assert Path("notebooks/reference/Huy-prediction-on-hospital-readmission.ipynb").is_file()
    assert Path("models/production_huy/model.pkl").is_file()
    assert not Path("models/production_v1").exists()
