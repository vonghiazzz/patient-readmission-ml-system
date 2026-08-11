from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import (
    IDENTIFIER_COLUMNS,
    TARGET_COLUMNS,
    create_preprocessor,
    fit_and_save_preprocessor,
    prepare_features,
)


def to_dense(matrix: object) -> np.ndarray:
    """Convert dense or sparse transformed output to a NumPy array."""
    toarray = getattr(matrix, "toarray", None)

    if callable(toarray):
        return np.asarray(toarray())

    return np.asarray(matrix)


@pytest.fixture
def train_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3],
            "patient_nbr": [101, 102, 103],
            "readmitted": ["NO", "<30", ">30"],
            "readmitted_30d": [0, 1, 0],
            "race": [
                "Caucasian",
                "AfricanAmerican",
                "Caucasian",
            ],
            "gender": [
                "Female",
                "Male",
                "Female",
            ],
            "age": [
                "[50-60)",
                "[60-70)",
                "[70-80)",
            ],
            "time_in_hospital": [2, 4, 6],
            "num_lab_procedures": [10, 20, 30],
            "num_procedures": [1, 2, 3],
            "num_medications": [10, 20, 30],
            "number_outpatient": [0, 1, 0],
            "number_emergency": [0, 0, 1],
            "number_inpatient": [0, 1, 2],
            "number_diagnoses": [4, 6, 8],
        }
    )


@pytest.fixture
def validation_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [4],
            "patient_nbr": [104],
            "readmitted": ["NO"],
            "readmitted_30d": [0],
            "race": ["NeverSeenRace"],
            "gender": ["Female"],
            "age": ["[80-90)"],
            "time_in_hospital": [5],
            "num_lab_procedures": [25],
            "num_procedures": [2],
            "num_medications": [9999],
            "number_outpatient": [1],
            "number_emergency": [1],
            "number_inpatient": [1],
            "number_diagnoses": [7],
        }
    )


def test_prepare_features_excludes_targets_and_identifiers(
    train_dataframe: pd.DataFrame,
) -> None:
    prepared = prepare_features(train_dataframe)

    excluded_columns = TARGET_COLUMNS | IDENTIFIER_COLUMNS

    assert excluded_columns.isdisjoint(prepared.columns)

    assert prepared["has_outpatient_history"].tolist() == [0, 1, 0]
    assert prepared["has_emergency_history"].tolist() == [0, 0, 1]
    assert prepared["has_inpatient_history"].tolist() == [0, 1, 1]

    # prepare_features must not mutate the original DataFrame.
    assert "readmitted" in train_dataframe.columns
    assert "patient_nbr" in train_dataframe.columns


def test_unknown_category_does_not_crash(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> None:
    train_features = prepare_features(train_dataframe)
    validation_features = prepare_features(validation_dataframe)

    preprocessor, _, _ = create_preprocessor(train_features)
    preprocessor.fit(train_features)

    transformed = preprocessor.transform(validation_features)
    dense_output = to_dense(transformed)

    assert dense_output.shape[0] == len(validation_dataframe)
    assert np.isfinite(dense_output).all()


def test_serialized_preprocessor_matches_original(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_features = prepare_features(train_dataframe)
    validation_features = prepare_features(validation_dataframe)

    preprocessor, _, _ = create_preprocessor(train_features)
    preprocessor.fit(train_features)

    output_before_save = to_dense(preprocessor.transform(validation_features))

    artifact_path = tmp_path / "preprocessor.joblib"
    joblib.dump(preprocessor, artifact_path)

    reloaded_preprocessor = joblib.load(artifact_path)

    output_after_reload = to_dense(reloaded_preprocessor.transform(validation_features))

    np.testing.assert_allclose(
        output_before_save,
        output_after_reload,
    )


def test_fit_and_save_records_metadata(
    train_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.csv"
    artifact_path = tmp_path / "preprocessor.joblib"
    metadata_path = tmp_path / "preprocessor_metadata.json"

    train_dataframe.to_csv(
        train_path,
        index=False,
    )

    fit_and_save_preprocessor(
        train_path=train_path,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )

    assert artifact_path.exists()
    assert artifact_path.stat().st_size > 0
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected_features = prepare_features(train_dataframe).columns.tolist()

    assert metadata["preprocessor_version"] == "v1"
    assert metadata["input_feature_order"] == expected_features
    assert metadata["target_columns_excluded"] == sorted(TARGET_COLUMNS)
    assert metadata["identifier_columns_excluded"] == sorted(IDENTIFIER_COLUMNS)

    assert set(metadata["numeric_columns"]).isdisjoint(TARGET_COLUMNS | IDENTIFIER_COLUMNS)
    assert set(metadata["categorical_columns"]).isdisjoint(TARGET_COLUMNS | IDENTIFIER_COLUMNS)


def test_preprocessor_is_fit_from_train_only(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.csv"
    artifact_path = tmp_path / "preprocessor.joblib"
    metadata_path = tmp_path / "preprocessor_metadata.json"

    train_dataframe.to_csv(
        train_path,
        index=False,
    )

    fit_and_save_preprocessor(
        train_path=train_path,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )

    preprocessor = joblib.load(artifact_path)

    numeric_columns = list(preprocessor.transformers_[0][2])
    categorical_columns = list(preprocessor.transformers_[1][2])

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]

    scaler = numeric_pipeline.named_steps["scaler"]
    encoder = categorical_pipeline.named_steps["encoder"]

    medication_index = numeric_columns.index("num_medications")
    race_index = categorical_columns.index("race")

    # Mean must come only from training values: [10, 20, 30].
    assert scaler.mean_[medication_index] == pytest.approx(20.0)

    race_categories_before = set(encoder.categories_[race_index].tolist())

    assert "NeverSeenRace" not in race_categories_before

    validation_features = prepare_features(validation_dataframe)

    # Validation uses transform only and must not change fitted state.
    transformed = preprocessor.transform(validation_features)

    assert transformed.shape[0] == 1
    assert scaler.mean_[medication_index] == pytest.approx(20.0)

    race_categories_after = set(encoder.categories_[race_index].tolist())

    assert race_categories_after == race_categories_before
    assert "NeverSeenRace" not in race_categories_after
