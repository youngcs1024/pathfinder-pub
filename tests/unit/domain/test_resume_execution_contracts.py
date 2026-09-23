from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.domain.errors import DomainValidationError
from app.domain.run_payloads import (
    EXECUTION_CONTRACTS,
    LEGACY_READ_CONTRACTS,
    ResumeRunInputV1,
    ResumeRunOutputV1,
    RunContractV1,
    RunMode,
    find_run_contract,
)
from app.domain.runs import RunStatus, create_request_digest_v1
from app.worker.contracts import RunExecutionResult


class SyntheticPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    receipt_id: UUID


@pytest.fixture
def contract():
    return RunContractV1(
        RunMode.MATERIAL_PREPARATION,
        "pathfinder-resume-v2",
        ResumeRunInputV1[SyntheticPayload],
        (ResumeRunOutputV1[SyntheticPayload],),
    )


def test_typed_resume_roundtrip_is_not_research_or_an_arbitrary_dictionary(contract):
    raw = {
        "schema_version": 1,
        "mode": "material_preparation",
        "payload": {"receipt_id": str(uuid4())},
    }
    request = contract.decode_input(raw)
    output = contract.decode_output(raw)
    assert isinstance(request.payload.receipt_id, UUID)
    assert output.model_dump(mode="json") == raw
    assert RunExecutionResult(status=RunStatus.COMPLETED, result=output).result == output
    registered = find_run_contract(EXECUTION_CONTRACTS, contract.graph_version, contract.mode)
    assert registered.input_model.__name__.startswith("MaterialPreparationRunInputV1")
    with pytest.raises(ValueError, match="unsupported"):
        find_run_contract(EXECUTION_CONTRACTS, "pathfinder-resume-v1", contract.mode)


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"mode": "resume_revision"},
        {"mode": "research"},
        {"actor_user_id": "untrusted"},
        {"payload": {"extra": "secret"}},
        {"payload": {"receipt_id": "not-a-uuid"}},
    ],
)
def test_input_and_output_reject_wrong_schema_mode_and_payload(contract, mutation):
    raw = {
        "schema_version": 1,
        "mode": "material_preparation",
        "payload": {"receipt_id": str(uuid4())},
        **mutation,
    }
    for decode in (contract.decode_input, contract.decode_output):
        with pytest.raises((ValueError, ValidationError)):
            decode(raw)


def test_unbound_or_mutable_payload_is_not_a_valid_result():
    class MutablePayload(BaseModel):
        number: int

    with pytest.raises(ValidationError):
        ResumeRunOutputV1[MutablePayload](
            mode=RunMode.MATERIAL_PREPARATION, payload=MutablePayload(number=1)
        )
    with pytest.raises(ValidationError):
        ResumeRunOutputV1(
            mode=RunMode.MATERIAL_PREPARATION, payload=SyntheticPayload(receipt_id=uuid4())
        )
    with pytest.raises(ValueError):
        RunExecutionResult(status=RunStatus.COMPLETED, result={"schema_version": 1})


@pytest.mark.parametrize("version", range(1, 7))
@pytest.mark.parametrize("schema_version", [1, 2])
def test_historical_outputs_keep_their_wire_format(version, schema_version):
    contract = find_run_contract(
        LEGACY_READ_CONTRACTS, f"pathfinder-research-v{version}", RunMode.RESEARCH
    )
    raw = {
        "schema_version": schema_version,
        "evidence_sufficient": False,
        "limitations": [{"code": "insufficient_evidence", "detail": "Synthetic missing evidence"}],
    }
    result = contract.decode_output(raw)
    assert result.schema_version == schema_version
    assert "mode" not in result.model_dump(mode="json")


@pytest.mark.parametrize(
    "mode", [RunMode.MATERIAL_PREPARATION, RunMode.RESUME_GENERATION, RunMode.RESUME_REVISION]
)
def test_new_modes_cannot_reuse_legacy_request_identity(mode):
    with pytest.raises(DomainValidationError):
        create_request_digest_v1(mode=mode, query="synthetic")


def test_constructed_invalid_resume_result_is_rejected_before_publication():
    output = ResumeRunOutputV1[SyntheticPayload](
        mode=RunMode.MATERIAL_PREPARATION, payload=SyntheticPayload(receipt_id=uuid4())
    ).model_copy(update={"schema_version": 2})
    with pytest.raises(ValueError, match="schema is invalid"):
        RunExecutionResult(status=RunStatus.COMPLETED, result=output)
