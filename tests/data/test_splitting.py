import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.splitting import (
    _hash_identifiers,
    _split_summary,
    create_huy_cohort,
    create_huy_splits,
    run_splitting,
)


def state() -> dict:
    return json.loads(
        Path("models/production_huy/preprocessing_state.json").read_text(encoding="utf-8")
    )


def base_record() -> dict:
    value = json.loads(Path("docs/api/sample_request.json").read_text(encoding="utf-8"))
    value.update(
        {
            "time_in_hospital": 4,
            "num_lab_procedures": 43,
            "num_procedures": 1,
            "num_medications": 16,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
            "number_diagnoses": 7,
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "diag_2": "401",
            "diag_3": "250",
        }
    )
    return value


def fixture_frame() -> pd.DataFrame:
    rows = []
    for index in range(20):
        row = {
            **base_record(),
            "encounter_id": 1000 + index,
            "patient_nbr": 2000 + index,
            "readmitted": "<30" if index % 2 else "NO",
        }
        rows.append(row)
    rows.append(
        {
            **base_record(),
            "encounter_id": 9999,
            "patient_nbr": 2000,
            "readmitted": "<30",
        }
    )
    return pd.DataFrame(rows)


def test_huy_cohort_keeps_first_encounter_and_maps_target() -> None:
    cohort = create_huy_cohort(fixture_frame(), state())
    assert len(cohort) == 20
    assert cohort["patient_nbr"].is_unique
    assert cohort.loc[cohort["patient_nbr"].eq(2000), "encounter_id"].item() == 1000
    assert set(cohort["readmitted_30d"]) == {0, 1}


def test_huy_split_is_stratified_80_20_without_patient_overlap() -> None:
    cohort = create_huy_cohort(fixture_frame(), state())
    train, test = create_huy_splits(cohort)
    assert len(train) == 16
    assert len(test) == 4
    assert set(train["patient_nbr"]).isdisjoint(test["patient_nbr"])
    assert train["readmitted_30d"].mean() == test["readmitted_30d"].mean() == 0.5


def test_create_huy_cohort_fails_when_required_column_is_missing() -> None:
    dataframe = fixture_frame().drop(columns=["diag_2"])

    with pytest.raises(
        ValueError,
        match="Huy cohort input is missing columns",
    ):
        create_huy_cohort(dataframe, state())


def test_create_huy_cohort_fails_on_unsupported_readmitted_label() -> None:
    dataframe = fixture_frame().copy()
    dataframe.loc[0, "readmitted"] = "INVALID"

    with pytest.raises(
        ValueError,
        match="unsupported readmitted label",
    ):
        create_huy_cohort(dataframe, state())


def test_create_huy_cohort_filters_invalid_rows() -> None:
    dataframe = fixture_frame().copy()

    invalid_encounters = {
        dataframe.loc[0, "encounter_id"],
        dataframe.loc[1, "encounter_id"],
        dataframe.loc[2, "encounter_id"],
    }

    dataframe.loc[0, "diag_1"] = "?"
    dataframe.loc[1, "gender"] = "Unknown/Invalid"
    dataframe.loc[2, "discharge_disposition_id"] = 11

    cohort = create_huy_cohort(dataframe, state())

    assert invalid_encounters.isdisjoint(set(cohort["encounter_id"]))
    assert "?" not in cohort["diag_1"].tolist()
    assert "Unknown/Invalid" not in cohort["gender"].tolist()
    assert 11 not in cohort["discharge_disposition_id"].tolist()


def test_hash_identifiers_is_deterministic() -> None:
    values = pd.Series([100, 200, 300])

    first = _hash_identifiers(values)
    second = _hash_identifiers(values)

    assert first == second
    assert len(first) == 64


def test_split_summary_returns_expected_counts() -> None:
    dataframe = pd.DataFrame(
        {
            "patient_nbr": [10, 11, 12, 13],
            "readmitted_30d": [0, 1, 0, 1],
        }
    )

    result = _split_summary(dataframe)

    assert result["row_count"] == 4
    assert result["patient_count"] == 4
    assert result["negative_count"] == 2
    assert result["positive_count"] == 2
    assert result["positive_rate"] == pytest.approx(0.5)
    assert len(result["patient_id_hash"]) == 64


def test_run_splitting_writes_outputs_and_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_directory = tmp_path / "splits"
    manifest_path = tmp_path / "manifest.json"
    preprocessing_state_path = Path("models/production_huy/preprocessing_state.json")

    input_dataframe = fixture_frame().copy()

    # Keep diagnosis columns as strings when pandas reads the CSV.
    # This mirrors the raw dataset, where "?" is also a valid missing marker.
    duplicate_row = input_dataframe["encounter_id"].eq(9999)
    input_dataframe.loc[
        duplicate_row,
        ["diag_1", "diag_2", "diag_3"],
    ] = "?"

    input_dataframe.to_csv(input_path, index=False)

    run_splitting(
        input_path=input_path,
        preprocessing_state_path=preprocessing_state_path,
        output_directory=output_directory,
        manifest_path=manifest_path,
        random_state=42,
    )

    train_path = output_directory / "train.csv"
    test_path = output_directory / "test.csv"

    assert train_path.exists()
    assert test_path.exists()
    assert manifest_path.exists()

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(train) == 16
    assert len(test) == 4

    assert set(train["patient_nbr"]).isdisjoint(set(test["patient_nbr"]))

    assert manifest["strategy"] == "huy_first_encounter_stratified_split"
    assert manifest["random_state"] == 42
    assert manifest["ratios"] == {"train": 0.8, "test": 0.2}
    assert manifest["target"] == "readmitted_30d"
    assert manifest["cohort_row_count"] == 20

    assert manifest["splits"]["train"]["row_count"] == 16
    assert manifest["splits"]["test"]["row_count"] == 4
