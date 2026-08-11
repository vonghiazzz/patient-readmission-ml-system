from __future__ import annotations

import pandas as pd

HISTORY_SOURCE_COLUMNS = {
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
}

DERIVED_FEATURES = (
    "has_outpatient_history",
    "has_emergency_history",
    "has_inpatient_history",
)


def build_v1_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the three deterministic features used by frozen feature set V1.

    The function never mutates its input and deliberately adds no experimental
    aggregates. Missing source columns are a contract error rather than being
    silently ignored.
    """

    missing = sorted(HISTORY_SOURCE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing V1 history source columns: {', '.join(missing)}")

    result = frame.copy()
    result["has_outpatient_history"] = (result["number_outpatient"] > 0).astype("int8")
    result["has_emergency_history"] = (result["number_emergency"] > 0).astype("int8")
    result["has_inpatient_history"] = (result["number_inpatient"] > 0).astype("int8")
    return result


# Backward-compatible public entry point for existing offline callers. There is
# still only one feature-engineering implementation.
def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    return build_v1_features(dataframe)
