import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import (
    CATEGORICAL_MODEL_FEATURES,
    MODEL_INPUT_FEATURES,
    build_huy_features,
)


def state() -> dict:
    return json.loads(
        Path("models/production_huy/preprocessing_state.json").read_text(encoding="utf-8")
    )


def payload() -> dict:
    return json.loads(Path("docs/api/sample_request.json").read_text(encoding="utf-8"))


def test_huy_preprocessing_is_deterministic_and_ordered() -> None:
    frame = pd.DataFrame([payload()])
    first = build_huy_features(frame, state())
    second = build_huy_features(frame, state())
    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == MODEL_INPUT_FEATURES
    assert first.shape == (1, 52)


def test_huy_manual_mappings_and_derived_features() -> None:
    transformed = build_huy_features(pd.DataFrame([payload()]), state()).iloc[0]
    assert transformed["gender"] == 0
    assert transformed["admission_type_id"] == "1"
    assert transformed["level1_diag1"] == "4"
    assert transformed["max_glu_serum"] == "1"
    assert transformed["A1Cresult"] == "1"
    assert transformed["insulin"] == 1
    assert transformed["metformin"] == 0
    assert np.isfinite(
        transformed.drop(labels=list(CATEGORICAL_MODEL_FEATURES)).astype(float).to_numpy()
    ).all()


def test_not_tested_lab_values_become_unknown_category() -> None:
    value = payload()
    value["max_glu_serum"] = None
    value["A1Cresult"] = "None"
    transformed = build_huy_features(pd.DataFrame([value]), state()).iloc[0]
    assert transformed["max_glu_serum"] == "Unknown"
    assert transformed["A1Cresult"] == "Unknown"
