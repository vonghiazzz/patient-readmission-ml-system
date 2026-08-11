"""Shared evaluation-data adaptation for the frozen V1 feature contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.features.build_features import build_v1_features


def _normalized_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def align_source_features(
    frame: pd.DataFrame,
    feature_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    """Align legacy snake_case CSV columns to exact manifest spellings.

    Matching is case/separator-insensitive and must be unique. This lets the
    repository's offline split files feed the frozen artifact without changing
    the public API contract.
    """

    lookup: dict[str, list[str]] = {}
    for column in frame.columns:
        lookup.setdefault(_normalized_name(str(column)), []).append(str(column))

    renamed: dict[str, str] = {}
    missing: list[str] = []
    for feature in feature_manifest["request_features"]:
        matches = lookup.get(_normalized_name(feature), [])
        if len(matches) == 1:
            renamed[matches[0]] = feature
        else:
            missing.append(feature)
    if missing:
        raise ValueError(f"Evaluation data cannot resolve V1 source features: {missing}")

    source = frame.rename(columns=renamed).loc[:, feature_manifest["request_features"]]
    model_input = build_v1_features(source)
    return model_input.loc[:, feature_manifest["model_input_features"]]
