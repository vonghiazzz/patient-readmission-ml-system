import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.api.schemas import PredictionRequest


def valid_payload() -> dict[str, object]:
    return {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "time_in_hospital": 4,
        "num_lab_procedures": 42,
        "num_procedures": 1,
        "num_medications": 12,
        "number_outpatient": 2,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 7,
        "insulin": "Steady",
    }


def test_prediction_request_accepts_documented_fields() -> None:
    request = PredictionRequest.model_validate(valid_payload())

    assert request.age == "[60-70)"
    assert request.number_inpatient == 1
    assert not hasattr(request, "patient_nbr")


@pytest.mark.parametrize(
    "field",
    [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ],
)
def test_prediction_request_rejects_invalid_counts(field: str) -> None:
    payload = valid_payload()
    payload[field] = -1

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


@pytest.mark.parametrize(
    "excluded_field",
    ["encounter_id", "patient_nbr", "readmitted", "readmitted_30d"],
)
def test_prediction_request_rejects_identifier_and_target_fields(
    excluded_field: str,
) -> None:
    payload = valid_payload()
    payload[excluded_field] = "sensitive-or-leaking-value"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PredictionRequest.model_validate(payload)


def test_openapi_schema_contains_example_and_excludes_leakage_fields() -> None:
    schema = PredictionRequest.model_json_schema()

    assert schema["examples"]
    assert schema["additionalProperties"] is False
    assert {
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "readmitted_30d",
        "service_utilization",
        "total_clinical_activities",
    }.isdisjoint(schema["properties"])


def test_request_fields_match_current_preprocessor_metadata() -> None:
    metadata_path = Path("models/preprocessor_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    engineered_fields = {"service_utilization", "total_clinical_activities"}
    expected_request_fields = set(metadata["input_feature_order"]) - engineered_fields

    assert set(PredictionRequest.model_fields) == expected_request_fields
