"""Create a privacy-safe, reproducible data handoff for the ML workstream."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.splitting import TARGET_MAPPING, assert_no_patient_overlap
from src.data.validation import load_quality_config, validate_dataframe
from src.features.preprocessing import prepare_features


def calculate_sha256(file_path: Path) -> str:
    """Calculate an artifact checksum without loading the full file into memory."""

    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(file_path: Path, description: str) -> None:
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {file_path}")


def verify_derived_target(dataframe: pd.DataFrame, split_name: str) -> None:
    required_columns = {"readmitted", "readmitted_30d"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{split_name} is missing target columns: {sorted(missing_columns)}")

    expected = dataframe["readmitted"].map(TARGET_MAPPING)
    if expected.isna().any():
        invalid_values = sorted(dataframe.loc[expected.isna(), "readmitted"].astype(str).unique())
        raise ValueError(f"{split_name} contains invalid target values: {invalid_values}")

    actual = pd.to_numeric(dataframe["readmitted_30d"], errors="coerce")
    if actual.isna().any() or not actual.astype("int8").equals(expected.astype("int8")):
        raise ValueError(f"{split_name} has inconsistent readmitted_30d values.")


def target_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    if "readmitted_30d" in dataframe.columns:
        target = pd.to_numeric(dataframe["readmitted_30d"], errors="raise").astype("int8")
    else:
        target = dataframe["readmitted"].map(TARGET_MAPPING)
        if target.isna().any():
            raise ValueError("Dataset contains target values outside the approved mapping.")
        target = target.astype("int8")

    counts = target.value_counts().reindex([0, 1], fill_value=0)
    total = int(len(target))
    return {
        "row_count": total,
        "negative_count": int(counts.loc[0]),
        "positive_count": int(counts.loc[1]),
        "positive_rate": float(counts.loc[1] / total) if total else 0.0,
    }


def split_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    return {
        **target_summary(dataframe),
        "patient_count": int(dataframe["patient_nbr"].nunique()),
    }


def missingness_summary(dataframe: pd.DataFrame, limit: int = 15) -> list[dict[str, Any]]:
    missing_count = dataframe.isna().sum()
    missing_rate = dataframe.isna().mean()
    ordered_columns = missing_rate.sort_values(ascending=False).index[:limit]
    return [
        {
            "column": str(column),
            "missing_count": int(missing_count[column]),
            "missing_rate": float(missing_rate[column]),
        }
        for column in ordered_columns
        if missing_count[column] > 0
    ]


def load_preprocessor_metadata(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Preprocessor metadata must be a JSON object.")
    return metadata


def verify_feature_contract(
    train: pd.DataFrame,
    preprocessor_path: Path,
    preprocessor_metadata_path: Path,
) -> dict[str, Any]:
    features = prepare_features(train)
    metadata = load_preprocessor_metadata(preprocessor_metadata_path)
    expected_feature_order = metadata.get("input_feature_order")

    if not isinstance(expected_feature_order, list):
        raise ValueError("Preprocessor metadata must contain input_feature_order.")

    actual_feature_order = features.columns.tolist()
    if actual_feature_order != expected_feature_order:
        missing = sorted(set(expected_feature_order) - set(actual_feature_order))
        unexpected = sorted(set(actual_feature_order) - set(expected_feature_order))
        raise ValueError(
            "Training features do not match preprocessor metadata: "
            f"missing={missing}, unexpected={unexpected}, order_matches=False"
        )

    preprocessor = joblib.load(preprocessor_path)
    sample_size = min(32, len(features))
    transformed = preprocessor.transform(features.head(sample_size))

    if transformed.shape[0] != sample_size:
        raise ValueError("Preprocessor changed the number of sample rows.")

    return {
        "preprocessor_version": metadata.get("preprocessor_version", "unknown"),
        "raw_input_feature_count": len(actual_feature_order),
        "transformed_feature_count": int(transformed.shape[1]),
        "input_feature_order": actual_feature_order,
        "numeric_columns": metadata.get("numeric_columns", []),
        "categorical_columns": metadata.get("categorical_columns", []),
        "target_columns_excluded": metadata.get("target_columns_excluded", []),
        "identifier_columns_excluded": metadata.get("identifier_columns_excluded", []),
        "sample_transform_verified": True,
    }


def build_handoff_report(
    ingested: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    quality_report: dict[str, Any],
    feature_contract: dict[str, Any],
    preprocessor_path: Path,
    preprocessor_metadata_path: Path,
) -> dict[str, Any]:
    return {
        "handoff_version": "v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Member A aggregate data handoff for Member B model development",
        "dataset": {
            "row_count": int(len(ingested)),
            "column_count": int(ingested.shape[1]),
            "patient_count": int(ingested["patient_nbr"].nunique()),
            "duplicate_encounter_count": int(
                ingested.duplicated(subset=["encounter_id"], keep="first").sum()
            ),
            "target": target_summary(ingested),
            "top_missing_columns": missingness_summary(ingested),
        },
        "data_quality": {
            "schema_passed": bool(quality_report["schema_passed"]),
            "errors": list(quality_report["errors"]),
            "warnings": list(quality_report["warnings"]),
        },
        "target_contract": {
            "name": "readmitted_30d",
            "mapping": TARGET_MAPPING,
            "positive_class": 1,
            "positive_meaning": "readmitted within 30 days (<30)",
        },
        "split_contract": {
            "strategy": "patient_aware_group_shuffle_split",
            "train": split_summary(train),
            "test": split_summary(test),
            "patient_overlap": False,
            "selection_policy": (
                "Use cross-validation or an inner validation split inside train only. "
                "Keep test locked until final evaluation."
            ),
        },
        "feature_contract": feature_contract,
        "artifacts": {
            "preprocessor_path": str(preprocessor_path),
            "preprocessor_sha256": calculate_sha256(preprocessor_path),
            "preprocessor_metadata_path": str(preprocessor_metadata_path),
            "preprocessor_metadata_sha256": calculate_sha256(preprocessor_metadata_path),
        },
        "privacy": {
            "contains_patient_level_rows": False,
            "patient_identifiers_published": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    target = dataset["target"]
    split = report["split_contract"]
    features = report["feature_contract"]

    missing_rows = [
        (f"| `{item['column']}` | {item['missing_count']} | {item['missing_rate']:.2%} |")
        for item in dataset["top_missing_columns"]
    ]
    if not missing_rows:
        missing_rows = ["| _None_ | 0 | 0.00% |"]

    return "\n".join(
        [
            "# Member A → Member B Data Handoff",
            "",
            f"Generated: `{report['generated_at_utc']}`",
            "",
            "## Dataset and target",
            "",
            f"- Rows: **{dataset['row_count']}**",
            f"- Columns: **{dataset['column_count']}**",
            f"- Unique patients: **{dataset['patient_count']}**",
            f"- Duplicate encounter IDs: **{dataset['duplicate_encounter_count']}**",
            f"- Positive rows (`<30`): **{target['positive_count']}** "
            f"(**{target['positive_rate']:.2%}**)",
            "- Target mapping: `<30 → 1`, `>30 → 0`, `NO → 0`",
            "",
            "## Patient-aware split",
            "",
            "| Split | Rows | Patients | Positive rate |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| Train | {split['train']['row_count']} | "
                f"{split['train']['patient_count']} | "
                f"{split['train']['positive_rate']:.2%} |"
            ),
            (
                f"| Test | {split['test']['row_count']} | "
                f"{split['test']['patient_count']} | "
                f"{split['test']['positive_rate']:.2%} |"
            ),
            "",
            "Patient overlap is verified as **zero**. Member B must use cross-validation or an "
            "inner validation split from `train` and keep `test` locked.",
            "",
            "## Feature and preprocessing contract",
            "",
            f"- Raw model inputs: **{features['raw_input_feature_count']}**",
            f"- Transformed features: **{features['transformed_feature_count']}**",
            f"- Numeric inputs: **{len(features['numeric_columns'])}**",
            f"- Categorical inputs: **{len(features['categorical_columns'])}**",
            f"- Preprocessor version: `{features['preprocessor_version']}`",
            f"- Preprocessor SHA-256: `{report['artifacts']['preprocessor_sha256']}`",
            "- Excluded targets: "
            + ", ".join(f"`{name}`" for name in features["target_columns_excluded"]),
            "- Excluded identifiers: "
            + ", ".join(f"`{name}`" for name in features["identifier_columns_excluded"]),
            "",
            "## Highest missing rates",
            "",
            "| Column | Missing rows | Missing rate |",
            "| --- | ---: | ---: |",
            *missing_rows,
            "",
            "## Handoff boundary",
            "",
            "Member A owns ingestion, validation, patient-aware splitting, feature construction "
            "and the fitted preprocessing artifact. Member B owns resampling inside training "
            "folds, baseline/model training, MLflow tracking, tuning and evaluation.",
            "",
            "This report contains aggregate statistics only and no patient-level records.",
            "",
        ]
    )


def atomic_write_text(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, output_path)


def run_handoff(
    input_path: Path,
    train_path: Path,
    test_path: Path,
    quality_config_path: Path,
    preprocessor_path: Path,
    preprocessor_metadata_path: Path,
    json_output_path: Path,
    markdown_output_path: Path,
) -> dict[str, Any]:
    for path, description in [
        (input_path, "Ingested dataset"),
        (train_path, "Training split"),
        (test_path, "Test split"),
        (quality_config_path, "Data quality config"),
        (preprocessor_path, "Fitted preprocessor"),
        (preprocessor_metadata_path, "Preprocessor metadata"),
    ]:
        require_file(path, description)

    ingested = pd.read_csv(input_path, low_memory=False)
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    quality_report = validate_dataframe(ingested, load_quality_config(quality_config_path))
    if not quality_report["schema_passed"]:
        formatted_errors = "; ".join(quality_report["errors"])
        raise ValueError(f"Data quality validation failed: {formatted_errors}")

    verify_derived_target(train, "train")
    verify_derived_target(test, "test")
    assert_no_patient_overlap(train, test)

    feature_contract = verify_feature_contract(
        train,
        preprocessor_path,
        preprocessor_metadata_path,
    )
    report = build_handoff_report(
        ingested,
        train,
        test,
        quality_report,
        feature_contract,
        preprocessor_path,
        preprocessor_metadata_path,
    )

    atomic_write_text(json_output_path, json.dumps(report, indent=2, ensure_ascii=False))
    atomic_write_text(markdown_output_path, render_markdown(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Member A to Member B data handoff.")
    parser.add_argument("--input", type=Path, default=Path("data/interim/ingested_data.csv"))
    parser.add_argument("--train", type=Path, default=Path("data/interim/splits/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/interim/splits/test.csv"))
    parser.add_argument("--quality-config", type=Path, default=Path("configs/data_quality.yaml"))
    parser.add_argument("--preprocessor", type=Path, default=Path("models/preprocessor.joblib"))
    parser.add_argument(
        "--preprocessor-metadata",
        type=Path,
        default=Path("models/preprocessor_metadata.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/member_a_ml_handoff.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/member_a_ml_handoff.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_handoff(
        input_path=args.input,
        train_path=args.train,
        test_path=args.test,
        quality_config_path=args.quality_config,
        preprocessor_path=args.preprocessor,
        preprocessor_metadata_path=args.preprocessor_metadata,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
    )
    print(
        "Member A handoff completed:",
        f"train_rows={report['split_contract']['train']['row_count']},",
        f"test_rows={report['split_contract']['test']['row_count']},",
        f"features={report['feature_contract']['raw_input_feature_count']}",
    )


if __name__ == "__main__":
    main()
