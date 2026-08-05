from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import build_features

TARGET_COLUMNS = {
    "readmitted",
    "readmitted_30d",
}

IDENTIFIER_COLUMNS = {
    "encounter_id",
    "patient_nbr",
}


def prepare_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    engineered = build_features(dataframe)

    columns_to_drop = [
        column for column in TARGET_COLUMNS | IDENTIFIER_COLUMNS if column in engineered.columns
    ]

    return engineered.drop(columns=columns_to_drop)


def create_preprocessor(
    dataframe: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in dataframe.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_columns, categorical_columns


def fit_and_save_preprocessor(
    train_path: Path,
    artifact_path: Path,
    metadata_path: Path,
) -> None:
    train = pd.read_csv(train_path, low_memory=False)
    train_features = prepare_features(train)

    preprocessor, numeric_columns, categorical_columns = create_preprocessor(train_features)

    # Fit only on training data.
    preprocessor.fit(train_features)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifact_path)

    metadata = {
        "preprocessor_version": "v1",
        "input_feature_order": train_features.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_columns_excluded": sorted(TARGET_COLUMNS),
        "identifier_columns_excluded": sorted(IDENTIFIER_COLUMNS),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit preprocessing pipeline.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/interim/splits/train.csv"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("models/preprocessor.joblib"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("models/preprocessor_metadata.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_and_save_preprocessor(
        train_path=args.train,
        artifact_path=args.artifact,
        metadata_path=args.metadata,
    )
    print("Preprocessor fitted and serialized successfully.")


if __name__ == "__main__":
    main()
