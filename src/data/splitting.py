"""Reproduce Huy's cohort and stratified 80/20 split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build_features import REQUEST_FEATURES, build_huy_features

TARGET_MAPPING = {"<30": 1, ">30": 0, "NO": 0}


def _hash_identifiers(values: pd.Series) -> str:
    normalized = "|".join(values.astype(str).tolist())
    return hashlib.sha256(normalized.encode()).hexdigest()


def create_huy_cohort(
    dataframe: pd.DataFrame,
    preprocessing_state: dict[str, Any],
) -> pd.DataFrame:
    """Apply the notebook's row filters, first-encounter policy, and outlier rules."""

    required = {
        "patient_nbr",
        "readmitted",
        "diag_2",
        "diag_3",
        *REQUEST_FEATURES,
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Huy cohort input is missing columns: {missing}")

    keep = (
        dataframe["diag_1"].ne("?")
        & dataframe["diag_2"].ne("?")
        & dataframe["diag_3"].ne("?")
        & dataframe["race"].ne("?")
        & dataframe["gender"].ne("Unknown/Invalid")
        & dataframe["discharge_disposition_id"].ne(11)
    )
    cohort = dataframe.loc[keep].drop_duplicates("patient_nbr", keep="first").copy()
    cohort["readmitted_30d"] = cohort["readmitted"].map(TARGET_MAPPING)
    if cohort["readmitted_30d"].isna().any():
        raise ValueError("Huy cohort contains an unsupported readmitted label")
    cohort["readmitted_30d"] = cohort["readmitted_30d"].astype("int8")

    raw_features = cohort.loc[:, REQUEST_FEATURES].copy()
    raw_features["max_glu_serum"] = raw_features["max_glu_serum"].fillna("None")
    raw_features["A1Cresult"] = raw_features["A1Cresult"].fillna("None")
    transformed = build_huy_features(raw_features, preprocessing_state)
    outlier_features = [
        *preprocessing_state["standard_1"]["features"],
        *preprocessing_state["standard_2"]["features"],
    ]
    keep_outliers = transformed.loc[:, outlier_features].abs().lt(3).all(axis=1)
    return cohort.loc[keep_outliers].copy()


def create_huy_splits(
    cohort: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        cohort,
        test_size=0.2,
        random_state=random_state,
        stratify=cohort["readmitted_30d"],
    )
    return train.copy(), test.copy()


def _split_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(dataframe)),
        "patient_count": int(dataframe["patient_nbr"].nunique()),
        "negative_count": int(dataframe["readmitted_30d"].eq(0).sum()),
        "positive_count": int(dataframe["readmitted_30d"].eq(1).sum()),
        "positive_rate": float(dataframe["readmitted_30d"].mean()),
        "patient_id_hash": _hash_identifiers(dataframe["patient_nbr"]),
    }


def run_splitting(
    input_path: Path,
    preprocessing_state_path: Path,
    output_directory: Path,
    manifest_path: Path,
    random_state: int = 42,
) -> None:
    dataframe = pd.read_csv(input_path, low_memory=False)
    state = json.loads(preprocessing_state_path.read_text(encoding="utf-8"))
    cohort = create_huy_cohort(dataframe, state)
    train, test = create_huy_splits(cohort, random_state=random_state)

    output_directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_directory / "train.csv", index=False)
    test.to_csv(output_directory / "test.csv", index=False)
    manifest = {
        "strategy": "huy_first_encounter_stratified_split",
        "random_state": random_state,
        "ratios": {"train": 0.8, "test": 0.2},
        "target": "readmitted_30d",
        "cohort_row_count": int(len(cohort)),
        "splits": {"train": _split_summary(train), "test": _split_summary(test)},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/diabetic_data.csv"))
    parser.add_argument(
        "--preprocessing-state",
        type=Path,
        default=Path("models/production_huy/preprocessing_state.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/splits"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/split_manifest.json"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_splitting(
        input_path=args.input,
        preprocessing_state_path=args.preprocessing_state,
        output_directory=args.output_dir,
        manifest_path=args.manifest,
        random_state=args.random_state,
    )
    print("Huy cohort and stratified train/test split completed.")


if __name__ == "__main__":
    main()
