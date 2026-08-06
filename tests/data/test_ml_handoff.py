from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from src.data.ml_handoff import run_handoff
from src.data.splitting import create_patient_aware_splits
from src.features.preprocessing import create_preprocessor, prepare_features


def create_fixture(tmp_path: Path) -> dict[str, Path]:
    dataframe = pd.DataFrame(
        {
            "encounter_id": list(range(1001, 1041)),
            "patient_nbr": list(range(1, 41)),
            "readmitted": ["<30", ">30", "NO", "NO"] * 10,
            "race": ["Caucasian", "AfricanAmerican"] * 20,
            "gender": ["Female", "Male"] * 20,
            "age": ["[50-60)", "[60-70)"] * 20,
            "time_in_hospital": [2, 4, 3, 5] * 10,
            "num_lab_procedures": [10, 20, 30, 40] * 10,
            "num_procedures": [0, 1, 2, 1] * 10,
            "num_medications": [5, 10, 15, 20] * 10,
            "number_outpatient": [0, 1, 0, 2] * 10,
            "number_emergency": [0, 0, 1, 0] * 10,
            "number_inpatient": [0, 1, 0, 2] * 10,
            "number_diagnoses": [3, 4, 5, 6] * 10,
        }
    )
    train, test = create_patient_aware_splits(dataframe)

    input_path = tmp_path / "ingested.csv"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    quality_config_path = tmp_path / "quality.yaml"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    metadata_path = tmp_path / "preprocessor_metadata.json"

    dataframe.to_csv(input_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    quality_config = {
        "required_columns": [
            "encounter_id",
            "patient_nbr",
            "readmitted",
            "gender",
            "age",
        ],
        "allowed_readmitted": ["<30", ">30", "NO"],
        "non_negative_columns": ["encounter_id", "patient_nbr", "time_in_hospital"],
        "duplicate_subset": ["encounter_id"],
        "max_duplicate_rows": 0,
        "missing_thresholds": {},
    }
    quality_config_path.write_text(yaml.safe_dump(quality_config), encoding="utf-8")

    train_features = prepare_features(train)
    preprocessor, numeric_columns, categorical_columns = create_preprocessor(train_features)
    preprocessor.fit(train_features)
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "preprocessor_version": "test-v1",
        "input_feature_order": train_features.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_columns_excluded": ["readmitted", "readmitted_30d"],
        "identifier_columns_excluded": ["encounter_id", "patient_nbr"],
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    return {
        "input_path": input_path,
        "train_path": train_path,
        "test_path": test_path,
        "quality_config_path": quality_config_path,
        "preprocessor_path": preprocessor_path,
        "preprocessor_metadata_path": metadata_path,
        "json_output_path": tmp_path / "handoff.json",
        "markdown_output_path": tmp_path / "handoff.md",
    }


def test_handoff_verifies_split_and_feature_contract(tmp_path: Path) -> None:
    paths = create_fixture(tmp_path)

    report = run_handoff(**paths)

    assert report["data_quality"]["schema_passed"] is True
    assert report["split_contract"]["patient_overlap"] is False
    assert report["split_contract"]["train"]["row_count"] == 28
    assert report["split_contract"]["test"]["row_count"] == 12
    assert report["feature_contract"]["sample_transform_verified"] is True
    assert report["privacy"]["contains_patient_level_rows"] is False
    assert paths["json_output_path"].exists()
    assert paths["markdown_output_path"].exists()


def test_handoff_rejects_feature_metadata_mismatch(tmp_path: Path) -> None:
    paths = create_fixture(tmp_path)
    metadata_path = paths["preprocessor_metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input_feature_order"] = list(reversed(metadata["input_feature_order"]))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="order_matches=False"):
        run_handoff(**paths)
