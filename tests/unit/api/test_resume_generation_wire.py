"""HTTP JSON decoding must not turn valid strict domain input into a 422."""

from copy import deepcopy
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from app.api.schemas.resume_generation import SessionCreateV1
from app.domain.resume_generation import SessionCreateV1 as DomainSessionCreateV1


def payload():
    return {
        "profile_version_id": str(uuid4()),
        "preference_version": 1,
        "project_ids": [str(uuid4())],
        "job": {"source": "paste", "text": "Synthetic JD"},
        "budget": {"max_model_calls": 3, "max_tool_calls": 0, "max_cost_cny": "1"},
    }


def test_http_json_retains_exact_domain_type_and_canonical_identity():
    assert TypeAdapter(SessionCreateV1).json_schema() == DomainSessionCreateV1.model_json_schema()
    body = payload()
    parsed = TypeAdapter(SessionCreateV1).validate_python(body)
    assert type(parsed) is DomainSessionCreateV1
    assert isinstance(parsed.profile_version_id, UUID)
    assert isinstance(parsed.project_ids, tuple)
    assert parsed.budget.max_cost_cny == Decimal("1")
    assert DomainSessionCreateV1.model_validate_json(parsed.model_dump_json()) == parsed
    with pytest.raises(ValidationError):
        DomainSessionCreateV1.model_validate(body)


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile_version_id", "invalid"),
        ("preference_version", "1"),
        ("preference_version", True),
        ("project_ids", []),
        ("project_ids", "invalid"),
        ("unexpected", "untrusted"),
    ],
)
def test_invalid_wire_values_remain_rejected(field, value):
    body = payload()
    body[field] = value
    with pytest.raises(ValidationError):
        TypeAdapter(SessionCreateV1).validate_python(body)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_model_calls", "3"),
        ("max_model_calls", True),
        ("max_cost_cny", "0"),
        ("max_cost_cny", "NaN"),
        ("max_tool_calls", 9),
        ("extra", 1),
    ],
)
def test_nested_budget_remains_strict(field, value):
    body = deepcopy(payload())
    body["budget"][field] = value
    with pytest.raises(ValidationError):
        TypeAdapter(SessionCreateV1).validate_python(body)
