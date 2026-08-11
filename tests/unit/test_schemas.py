import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.api.schemas import PredictionRequest


def manifest() -> dict:
    return json.loads(
        Path("models/production_v1/feature_manifest.json").read_text(encoding="utf-8")
    )


def minimal_payload() -> dict[str, object]:
    payload: dict[str, object] = {}
    numeric = {
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "admission_type_id",
        "admission_source_id",
    }
    for feature in manifest()["request_features"]:
        payload[feature] = 1 if feature in numeric else "No"
    payload.update({"race": "Caucasian", "gender": "Female", "age": "[60-70)"})
    return payload


def test_request_fields_match_production_manifest_exactly() -> None:
    assert list(PredictionRequest.model_fields) == manifest()["request_features"]


def test_prediction_request_accepts_all_and_only_manifest_fields() -> None:
    request = PredictionRequest.model_validate(minimal_payload())
    assert request.model_dump() == minimal_payload()

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PredictionRequest.model_validate({**minimal_payload(), "patient_nbr": 123})


def test_categorical_ids_preserve_integer_input_semantics() -> None:
    payload = minimal_payload()
    payload["admission_type_id"] = "not-an-integer"
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


def test_missing_nullable_category_is_allowed_but_omission_is_not() -> None:
    payload = minimal_payload()
    payload["race"] = None
    assert PredictionRequest.model_validate(payload).model_dump()["race"] is None

    del payload["race"]
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


def test_openapi_schema_has_exact_42_properties_and_forbids_extras() -> None:
    schema = PredictionRequest.model_json_schema()
    assert list(schema["properties"]) == manifest()["request_features"]
    assert len(schema["required"]) == 42
    assert schema["additionalProperties"] is False
    assert set(schema["example"]) == set(manifest()["request_features"])
    assert "patient_nbr" not in schema["example"]
    assert "encounter_id" not in schema["example"]
