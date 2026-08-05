from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


TARGET_MAPPING = {
    "<30": 1,
    ">30": 0,
    "NO": 0,
}


def hash_identifiers(values: pd.Series) -> str:
    normalized = "|".join(
        sorted(values.dropna().astype(str).unique().tolist())
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def add_binary_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result["readmitted_30d"] = result["readmitted"].map(TARGET_MAPPING)

    if result["readmitted_30d"].isna().any():
        invalid = sorted(
            result.loc[
                result["readmitted_30d"].isna(),
                "readmitted",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"Unable to map target values: {invalid}")

    result["readmitted_30d"] = result["readmitted_30d"].astype("int8")
    return result


def create_patient_aware_splits(
    dataframe: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"patient_nbr", "readmitted"}
    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    dataframe = add_binary_target(dataframe)

    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=random_state,
    )

    train_index, temporary_index = next(
        first_split.split(
            dataframe,
            y=dataframe["readmitted_30d"],
            groups=dataframe["patient_nbr"],
        )
    )

    train = dataframe.iloc[train_index].copy()
    temporary = dataframe.iloc[temporary_index].copy()

    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=0.50,
        random_state=random_state,
    )

    validation_index, test_index = next(
        second_split.split(
            temporary,
            y=temporary["readmitted_30d"],
            groups=temporary["patient_nbr"],
        )
    )

    validation = temporary.iloc[validation_index].copy()
    test = temporary.iloc[test_index].copy()

    return train, validation, test


def assert_no_patient_overlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    train_patients = set(train["patient_nbr"])
    validation_patients = set(validation["patient_nbr"])
    test_patients = set(test["patient_nbr"])

    if train_patients & validation_patients:
        raise ValueError("Patient overlap detected between train and validation.")

    if train_patients & test_patients:
        raise ValueError("Patient overlap detected between train and test.")

    if validation_patients & test_patients:
        raise ValueError("Patient overlap detected between validation and test.")


def split_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    return {
        "row_count": int(len(dataframe)),
        "patient_count": int(dataframe["patient_nbr"].nunique()),
        "positive_rate": float(dataframe["readmitted_30d"].mean()),
        "patient_id_hash": hash_identifiers(dataframe["patient_nbr"]),
    }


def run_splitting(
    input_path: Path,
    output_directory: Path,
    manifest_path: Path,
    random_state: int = 42,
) -> None:
    dataframe = pd.read_csv(input_path, low_memory=False)

    train, validation, test = create_patient_aware_splits(
        dataframe,
        random_state=random_state,
    )
    assert_no_patient_overlap(train, validation, test)

    output_directory.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_directory / "train.csv", index=False)
    validation.to_csv(output_directory / "validation.csv", index=False)
    test.to_csv(output_directory / "test.csv", index=False)

    manifest = {
        "strategy": "patient_aware_group_shuffle_split",
        "random_state": random_state,
        "ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "target": "readmitted_30d",
        "splits": {
            "train": split_summary(train),
            "validation": split_summary(validation),
            "test": split_summary(test),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-safe splits.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/interim/ingested_data.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/splits"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/split_manifest.json"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_splitting(
        input_path=args.input,
        output_directory=args.output_dir,
        manifest_path=args.manifest,
        random_state=args.random_state,
    )
    print("Train/validation/test split completed.")


if __name__ == "__main__":
    main()