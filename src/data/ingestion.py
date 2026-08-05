from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 checksum without loading the entire file into memory."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized snake-case-like column names."""
    normalized = dataframe.copy()
    normalized.columns = [
        column.strip().lower().replace(" ", "_").replace("-", "_") for column in normalized.columns
    ]
    return normalized


def load_raw_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Raw dataset does not exist: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    try:
        dataframe = pd.read_csv(
            input_path,
            na_values=["?"],
            keep_default_na=True,
            low_memory=False,
        )
    except UnicodeDecodeError:
        dataframe = pd.read_csv(
            input_path,
            encoding="latin-1",
            na_values=["?"],
            keep_default_na=True,
            low_memory=False,
        )

    if dataframe.empty:
        raise ValueError("Raw dataset is empty.")

    return normalize_column_names(dataframe)


def atomic_write_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write through a temporary file so a failed run does not corrupt output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)


def write_metadata(
    dataframe: pd.DataFrame,
    input_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    checksum = calculate_sha256(input_path)

    metadata: dict[str, object] = {
        "dataset_name": "Diabetes 130-US Hospitals",
        "dataset_version": checksum[:12],
        "source_filename": input_path.name,
        "source_size_bytes": input_path.stat().st_size,
        "sha256": checksum,
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "columns": dataframe.columns.tolist(),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, metadata_path)

    return metadata


def run_ingestion(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    dataframe = load_raw_dataset(input_path)
    atomic_write_csv(dataframe, output_path)
    return write_metadata(dataframe, input_path, metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest hospital readmission data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/diabetic_data.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/ingested_data.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/metadata/ingestion_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_ingestion(args.input, args.output, args.metadata)

    print(
        "Ingestion completed:",
        f"rows={metadata['row_count']},",
        f"columns={metadata['column_count']},",
        f"version={metadata['dataset_version']}",
    )


if __name__ == "__main__":
    main()
