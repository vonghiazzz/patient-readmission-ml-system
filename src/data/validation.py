from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_quality_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Quality config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Data quality config must be a YAML mapping.")

    return config


def validate_dataframe(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    required_columns = set(config.get("required_columns", []))
    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    allowed_labels = set(config.get("allowed_readmitted", []))
    if "readmitted" in dataframe.columns and allowed_labels:
        actual_labels = set(dataframe["readmitted"].dropna().astype(str).unique())
        invalid_labels = sorted(actual_labels - allowed_labels)

        if invalid_labels:
            errors.append(f"Invalid readmitted values: {invalid_labels}")

    for column in config.get("non_negative_columns", []):
        if column not in dataframe.columns:
            continue

        numeric_values = pd.to_numeric(dataframe[column], errors="coerce")
        negative_count = int((numeric_values < 0).sum())

        if negative_count:
            errors.append(f"Column '{column}' contains {negative_count} negative values.")

    missing_rate = {
        column: float(rate)
        for column, rate in dataframe.isna().mean().sort_values(ascending=False).items()
    }

    for column, threshold in config.get("missing_thresholds", {}).items():
        if column not in dataframe.columns:
            continue

        actual_rate = missing_rate[column]
        if actual_rate > float(threshold):
            errors.append(
                f"Column '{column}' missing rate {actual_rate:.4f} "
                f"exceeds threshold {float(threshold):.4f}."
            )

    duplicate_subset = config.get("duplicate_subset", [])
    usable_subset = [column for column in duplicate_subset if column in dataframe.columns]

    duplicate_count = int(dataframe.duplicated(subset=usable_subset or None).sum())

    if duplicate_count:
        warnings.append(f"Detected {duplicate_count} duplicate rows/keys.")

    label_positive_rate = None
    if "readmitted" in dataframe.columns:
        binary_target = dataframe["readmitted"].map({"<30": 1, ">30": 0, "NO": 0})
        if binary_target.notna().any():
            label_positive_rate = float(binary_target.mean())

    return {
        "schema_passed": not errors,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "missing_rate": missing_rate,
        "duplicate_count": duplicate_count,
        "label_positive_rate": label_positive_rate,
        "errors": errors,
        "warnings": warnings,
    }


def write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_validation(
    input_path: Path,
    config_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    dataframe = pd.read_csv(input_path, low_memory=False)
    config = load_quality_config(config_path)
    report = validate_dataframe(dataframe, config)
    write_report(report, report_path)

    if not report["schema_passed"]:
        formatted_errors = "\n".join(f"- {error}" for error in report["errors"])
        raise ValueError(f"Data validation failed:\n{formatted_errors}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ingested data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/interim/ingested_data.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_quality.yaml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/validation_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_validation(args.input, args.config, args.report)

    print(
        "Validation passed:",
        f"rows={report['row_count']},",
        f"duplicates={report['duplicate_count']}",
    )


if __name__ == "__main__":
    main()
