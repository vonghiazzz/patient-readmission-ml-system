import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.api.schemas import PredictionRequest
from src.features.build_features import REQUEST_FEATURES


def payload() -> dict:
    return json.loads(Path("docs/api/sample_request.json").read_text(encoding="utf-8"))


def test_request_has_exact_huy_raw_fields_and_forbids_extras() -> None:
    schema = PredictionRequest.model_json_schema()
    assert tuple(PredictionRequest.model_fields) == REQUEST_FEATURES
    assert tuple(schema["properties"]) == REQUEST_FEATURES
    assert len(schema["required"]) == 40
    assert schema["additionalProperties"] is False
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({**payload(), "patient_nbr": 123})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("discharge_disposition_id", 11),
        ("time_in_hospital", 15),
        ("gender", "Unknown/Invalid"),
        ("diag_1", "?"),
        ("insulin", "Invalid"),
    ],
)
def test_notebook_excluded_or_out_of_contract_values_are_rejected(
    field: str, value: object
) -> None:
    invalid = payload()
    invalid[field] = value
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(invalid)


def test_not_tested_labs_accept_source_token_or_null() -> None:
    source_token = payload()
    source_token["max_glu_serum"] = "None"
    source_token["A1Cresult"] = "None"
    PredictionRequest.model_validate(source_token)

    null_token = payload()
    null_token["max_glu_serum"] = None
    null_token["A1Cresult"] = None
    PredictionRequest.model_validate(null_token)
