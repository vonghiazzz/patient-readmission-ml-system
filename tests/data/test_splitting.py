import pandas as pd

from src.data.splitting import (
    assert_no_patient_overlap,
    create_patient_aware_splits,
)


def test_patient_does_not_cross_splits() -> None:
    dataframe = pd.DataFrame(
        {
            "patient_nbr": list(range(1, 41)),
            "encounter_id": list(range(101, 141)),
            "readmitted": ["<30", ">30", "NO", "NO"] * 10,
        }
    )

    train, validation, test = create_patient_aware_splits(
        dataframe,
        random_state=42,
    )

    assert_no_patient_overlap(train, validation, test)
