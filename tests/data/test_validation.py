import pandas as pd

from src.data.validation import validate_dataframe


def test_invalid_target_fails_validation() -> None:
    dataframe = pd.DataFrame(
        {
            "encounter_id": [1],
            "patient_nbr": [10],
            "readmitted": ["INVALID"],
        }
    )

    config = {
        "required_columns": [
            "encounter_id",
            "patient_nbr",
            "readmitted",
        ],
        "allowed_readmitted": ["<30", ">30", "NO"],
        "non_negative_columns": [],
        "duplicate_subset": ["encounter_id"],
        "missing_thresholds": {},
    }

    report = validate_dataframe(dataframe, config)

    assert report["schema_passed"] is False
    assert report["errors"]
