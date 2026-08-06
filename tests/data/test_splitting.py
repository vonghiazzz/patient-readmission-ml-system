import json
from pathlib import Path

import pandas as pd

from src.data.splitting import (
    assert_no_patient_overlap,
    create_patient_aware_splits,
    run_splitting,
)


def test_patient_does_not_cross_splits() -> None:
    dataframe = pd.DataFrame(
        {
            "patient_nbr": list(range(1, 41)),
            "encounter_id": list(range(101, 141)),
            "readmitted": ["<30", ">30", "NO", "NO"] * 10,
        }
    )

    train, test = create_patient_aware_splits(
        dataframe,
        random_state=42,
    )

    assert_no_patient_overlap(train, test)
    assert len(train) + len(test) == len(dataframe)


def test_run_splitting_writes_only_train_and_test(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "patient_nbr": list(range(1, 41)),
            "encounter_id": list(range(101, 141)),
            "readmitted": ["<30", ">30", "NO", "NO"] * 10,
        }
    )
    input_path = tmp_path / "input.csv"
    output_directory = tmp_path / "splits"
    manifest_path = tmp_path / "split_manifest.json"

    dataframe.to_csv(input_path, index=False)
    output_directory.mkdir()
    (output_directory / "validation.csv").write_text("legacy", encoding="utf-8")

    run_splitting(input_path, output_directory, manifest_path)

    assert sorted(path.name for path in output_directory.iterdir()) == ["test.csv", "train.csv"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["ratios"] == {"train": 0.70, "test": 0.30}
    assert set(manifest["splits"]) == {"train", "test"}
