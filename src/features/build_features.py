from __future__ import annotations

import pandas as pd

REQUIRED_SERVICE_COLUMNS = {
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
}

REQUIRED_ACTIVITY_COLUMNS = {
    "num_lab_procedures",
    "num_procedures",
}


def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    if REQUIRED_SERVICE_COLUMNS.issubset(result.columns):
        result["service_utilization"] = (
            result["number_outpatient"].fillna(0)
            + result["number_emergency"].fillna(0)
            + result["number_inpatient"].fillna(0)
        )

    if REQUIRED_ACTIVITY_COLUMNS.issubset(result.columns):
        result["total_clinical_activities"] = result["num_lab_procedures"].fillna(0) + result[
            "num_procedures"
        ].fillna(0)

    return result
