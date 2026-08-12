import json
from pathlib import Path

import pandas as pd

from src.data.splitting import create_huy_cohort, create_huy_splits


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
