"""Align raw evaluation data to Huy's exact request and model contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.features.build_features import REQUEST_FEATURES, build_huy_features


def _normalized_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def align_source_features(
    frame: pd.DataFrame,
    preprocessing_state: Mapping[str, Any],
) -> pd.DataFrame:
    lookup: dict[str, list[str]] = {}
    for column in frame.columns:
        lookup.setdefault(_normalized_name(str(column)), []).append(str(column))

    renamed: dict[str, str] = {}
    missing: list[str] = []
    for feature in REQUEST_FEATURES:
        matches = lookup.get(_normalized_name(feature), [])
        if len(matches) == 1:
            renamed[matches[0]] = feature
        else:
            missing.append(feature)
    if missing:
        raise ValueError(f"Evaluation data cannot resolve Huy raw features: {missing}")

    source = frame.rename(columns=renamed).loc[:, REQUEST_FEATURES].copy()
    source["max_glu_serum"] = source["max_glu_serum"].fillna("None")
    source["A1Cresult"] = source["A1Cresult"].fillna("None")
    return build_huy_features(source, preprocessing_state)
