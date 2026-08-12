"""Exact raw-to-model feature construction used by Huy's final CatBoost model."""

from __future__ import annotations

from collections.abc import Mapping
from math import floor, log, log1p
from typing import Any

import pandas as pd

MEDICATION_FEATURES = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "insulin",
    "glyburide-metformin",
    "tolazamide",
    "metformin-pioglitazone",
    "metformin-rosiglitazone",
    "glimepiride-pioglitazone",
    "glipizide-metformin",
    "troglitazone",
    "tolbutamide",
    "acetohexamide",
)

REQUEST_FEATURES = (
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    *MEDICATION_FEATURES,
    "change",
    "diabetesMed",
    "diag_1",
)

MODEL_INPUT_FEATURES = (
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "numchange",
    "level1_diag1",
    "nummed",
    "number_emergency_log1p",
    "number_outpatient_log1p",
    "service_utilization_log1p",
    "number_inpatient_log1p",
    "time_in_hospital|num_lab_procedures",
    "num_medications|num_lab_procedures",
    "num_medications|number_diagnoses",
    "age|number_diagnoses",
    "change|num_medications",
    "number_diagnoses|time_in_hospital",
    "num_medications|time_in_hospital_log",
    "num_medications|num_procedures_log1p",
    "num_medications|numchange_log1p",
)

CATEGORICAL_MODEL_FEATURES = (
    "race",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "level1_diag1",
)

AGE_MIDPOINTS = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

ADMISSION_TYPE_MAPPING = {2: 1, 7: 1, 6: 5, 8: 5}
DISCHARGE_DISPOSITION_MAPPING = {
    6: 1,
    8: 1,
    9: 1,
    13: 1,
    3: 2,
    4: 2,
    5: 2,
    14: 2,
    22: 2,
    23: 2,
    24: 2,
    12: 10,
    15: 10,
    16: 10,
    17: 10,
    25: 18,
    26: 18,
}
ADMISSION_SOURCE_MAPPING = {
    2: 1,
    3: 1,
    5: 4,
    6: 4,
    10: 4,
    22: 4,
    25: 4,
    15: 9,
    17: 9,
    20: 9,
    21: 9,
    13: 11,
    14: 11,
}


def _diagnosis_group(value: str) -> int:
    """Collapse a primary ICD-9 code into the notebook's nine level-1 groups."""

    normalized = value.strip().upper()
    if normalized.startswith(("V", "E")):
        return 0

    code = float(normalized)
    if 390 <= code < 460 or floor(code) == 785:
        return 1
    if 460 <= code < 520 or floor(code) == 786:
        return 2
    if 520 <= code < 580 or floor(code) == 787:
        return 3
    if floor(code) == 250:
        return 4
    if 800 <= code < 1000:
        return 5
    if 710 <= code < 740:
        return 6
    if 580 <= code < 630 or floor(code) == 788:
        return 7
    if 140 <= code < 240:
        return 8
    return 0


def _lab_result(value: Any, abnormal_values: set[str]) -> str:
    if value is None or str(value) in {"None", "Unknown"}:
        return "Unknown"
    return "1" if str(value) in abnormal_values else "0"


def _standardize(frame: pd.DataFrame, section: Mapping[str, Any]) -> None:
    for feature in section["features"]:
        frame[feature] = (frame[feature].astype(float) - float(section["mean"][feature])) / float(
            section["scale"][feature]
        )


def _minmax(frame: pd.DataFrame, section: Mapping[str, Any]) -> None:
    for feature in section["features"]:
        minimum = float(section["min"][feature])
        maximum = float(section["max"][feature])
        frame[feature] = (frame[feature].astype(float) - minimum) / (maximum - minimum)


def build_huy_features(
    frame: pd.DataFrame,
    preprocessing_state: Mapping[str, Any],
) -> pd.DataFrame:
    """Transform raw encounter fields to the exact 52-column CatBoost contract.

    The fitted statistics are frozen from the notebook cohort. This function only
    applies them and never fits state from an inference request.
    """

    submitted = set(frame.columns)
    expected = set(REQUEST_FEATURES)
    if submitted != expected:
        missing = sorted(expected - submitted)
        extra = sorted(submitted - expected)
        raise ValueError(f"Huy request contract mismatch; missing={missing}, extra={extra}")

    result = frame.loc[:, REQUEST_FEATURES].copy()
    raw_time = result["time_in_hospital"].astype(float)
    raw_labs = result["num_lab_procedures"].astype(float)
    raw_procedures = result["num_procedures"].astype(float)
    raw_medication_count = result["num_medications"].astype(float)
    raw_diagnosis_count = result["number_diagnoses"].astype(float)

    result["gender"] = result["gender"].map({"Female": 0, "Male": 1})
    result["age"] = result["age"].map(AGE_MIDPOINTS).astype(float)
    result["change"] = result["change"].map({"No": 0, "Ch": 1})
    result["diabetesMed"] = result["diabetesMed"].map({"No": 0, "Yes": 1})

    result["admission_type_id"] = result["admission_type_id"].replace(ADMISSION_TYPE_MAPPING)
    result["discharge_disposition_id"] = result["discharge_disposition_id"].replace(
        DISCHARGE_DISPOSITION_MAPPING
    )
    result["admission_source_id"] = result["admission_source_id"].replace(ADMISSION_SOURCE_MAPPING)
    result["max_glu_serum"] = result["max_glu_serum"].map(
        lambda value: _lab_result(value, {">200", ">300"})
    )
    result["A1Cresult"] = result["A1Cresult"].map(lambda value: _lab_result(value, {">7", ">8"}))
    result["level1_diag1"] = result["diag_1"].map(_diagnosis_group)

    raw_medication_states = result.loc[:, MEDICATION_FEATURES].copy()
    result["numchange"] = raw_medication_states.isin(["Up", "Down"]).sum(axis=1)
    result["nummed"] = raw_medication_states.ne("No").sum(axis=1)
    for feature in MEDICATION_FEATURES:
        result[feature] = raw_medication_states[feature].ne("No").astype(int)

    service_utilization = (
        result["number_outpatient"].astype(float)
        + result["number_emergency"].astype(float)
        + result["number_inpatient"].astype(float)
    )
    result["number_emergency_log1p"] = result["number_emergency"].map(log1p)
    result["number_outpatient_log1p"] = result["number_outpatient"].map(log1p)
    result["service_utilization_log1p"] = service_utilization.map(log1p)
    result["number_inpatient_log1p"] = result["number_inpatient"].map(log1p)

    raw_age = result["age"].copy()
    raw_change = result["change"].astype(float)
    raw_numchange = result["numchange"].astype(float)
    result["time_in_hospital|num_lab_procedures"] = raw_time * raw_labs
    result["num_medications|num_lab_procedures"] = raw_medication_count * raw_labs
    result["num_medications|number_diagnoses"] = raw_medication_count * raw_diagnosis_count
    result["age|number_diagnoses"] = raw_age * raw_diagnosis_count
    result["change|num_medications"] = raw_change * raw_medication_count
    result["number_diagnoses|time_in_hospital"] = raw_diagnosis_count * raw_time
    result["num_medications|time_in_hospital_log"] = (raw_medication_count * raw_time).map(log)
    result["num_medications|num_procedures_log1p"] = (raw_medication_count * raw_procedures).map(
        log1p
    )
    result["num_medications|numchange_log1p"] = (raw_medication_count * raw_numchange).map(log1p)

    _standardize(result, preprocessing_state["standard_1"])
    _minmax(result, preprocessing_state["minmax_1"])
    _standardize(result, preprocessing_state["standard_2"])

    age_state = preprocessing_state["age_minmax"]
    result["age"] = (result["age"] - float(age_state["min"])) / (
        float(age_state["max"]) - float(age_state["min"])
    )

    for feature in CATEGORICAL_MODEL_FEATURES:
        result[feature] = result[feature].astype(str)

    return result.loc[:, MODEL_INPUT_FEATURES]


def build_features(
    dataframe: pd.DataFrame,
    preprocessing_state: Mapping[str, Any],
) -> pd.DataFrame:
    """Public feature-building entry point."""

    return build_huy_features(dataframe, preprocessing_state)
