import pandas as pd
import pytest

from src.data.validation import validate_dataframe


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [1, 2],
            "patient_nbr": [101, 102],
            "race": ["Caucasian", "AfricanAmerican"],
            "gender": ["Female", "Male"],
            "age": ["[50-60)", "[60-70)"],
            "time_in_hospital": [3, 5],
            "num_lab_procedures": [40, 50],
            "num_procedures": [1, 2],
            "num_medications": [10, 12],
            "number_outpatient": [0, 1],
            "number_emergency": [0, 0],
            "number_inpatient": [0, 1],
            "number_diagnoses": [5, 6],
            "readmitted": ["NO", "<30"],
        }
    )


@pytest.fixture
def quality_config() -> dict:
    return {
        "required_columns": [
            "encounter_id",
            "patient_nbr",
            "race",
            "gender",
            "age",
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
            "readmitted",
        ],
        "allowed_readmitted": ["<30", ">30", "NO"],
        "non_negative_columns": [
            "encounter_id",
            "patient_nbr",
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
        ],
        "duplicate_subset": ["encounter_id"],
        "max_duplicate_rows": 0,
        "missing_thresholds": {},
    }


def test_out_of_range_value_fails_validation(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    invalid_dataframe = valid_dataframe.copy()
    invalid_dataframe.loc[0, "time_in_hospital"] = 99

    config = {
        **quality_config,
        "value_ranges": {
            "time_in_hospital": {
                "min": 1,
                "max": 14,
            }
        },
    }

    assert_validation_failed(invalid_dataframe, config)


def assert_validation_failed(
    dataframe: pd.DataFrame,
    config: dict,
) -> None:
    report = validate_dataframe(dataframe, config)

    assert report["schema_passed"] is False
    assert report["errors"]


def test_valid_dataframe_passes_validation(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    report = validate_dataframe(valid_dataframe, quality_config)

    assert report["schema_passed"] is True
    assert report["errors"] == []


def test_invalid_target_fails_validation(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    invalid_dataframe = valid_dataframe.copy()
    invalid_dataframe.loc[0, "readmitted"] = "INVALID"

    assert_validation_failed(invalid_dataframe, quality_config)


def test_missing_required_column_fails_validation(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    invalid_dataframe = valid_dataframe.drop(columns=["patient_nbr"])

    assert_validation_failed(invalid_dataframe, quality_config)


def test_negative_value_fails_validation(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    invalid_dataframe = valid_dataframe.copy()
    invalid_dataframe.loc[0, "num_medications"] = -1

    assert_validation_failed(invalid_dataframe, quality_config)


def test_missing_threshold_violation_fails_validation(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    invalid_dataframe = valid_dataframe.copy()
    invalid_dataframe.loc[0, "race"] = None

    config = {
        **quality_config,
        "missing_thresholds": {
            "race": 0.0,
        },
    }

    assert_validation_failed(invalid_dataframe, config)


def test_duplicate_violation_fails_validation(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    invalid_dataframe = pd.concat(
        [valid_dataframe, valid_dataframe.iloc[[0]]],
        ignore_index=True,
    )

    assert_validation_failed(invalid_dataframe, quality_config)


def test_non_numeric_value_fails_validation(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    invalid_dataframe = valid_dataframe.copy()
    invalid_dataframe["num_medications"] = invalid_dataframe["num_medications"].astype(object)
    invalid_dataframe.loc[0, "num_medications"] = "invalid"

    report = validate_dataframe(invalid_dataframe, quality_config)

    assert report["schema_passed"] is False
    assert any("contains 1 non-numeric values" in error for error in report["errors"])


def test_invalid_range_configuration_fails_validation(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    config = {
        **quality_config,
        "value_ranges": {
            "time_in_hospital": "invalid-config",
        },
    }

    report = validate_dataframe(valid_dataframe, config)

    assert report["schema_passed"] is False
    assert (
        "Range configuration for column 'time_in_hospital' must be a mapping." in report["errors"]
    )


def test_missing_duplicate_column_fails_validation(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    config = {
        **quality_config,
        "duplicate_subset": ["nonexistent_column"],
    }

    report = validate_dataframe(valid_dataframe, config)

    assert report["schema_passed"] is False
    assert any("Duplicate check columns are missing" in error for error in report["errors"])


def test_allowed_duplicate_rows_generate_warning(
    valid_dataframe: pd.DataFrame,
    quality_config: dict,
) -> None:
    duplicated_dataframe = pd.concat(
        [valid_dataframe, valid_dataframe.iloc[[0]]],
        ignore_index=True,
    )

    config = {
        **quality_config,
        "max_duplicate_rows": 1,
    }

    report = validate_dataframe(duplicated_dataframe, config)

    assert report["schema_passed"] is True
    assert report["duplicate_count"] == 1
    assert any("Detected 1 duplicate rows/keys" in warning for warning in report["warnings"])
