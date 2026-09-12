from pydantic import Field, field_validator, model_validator

from app.domain.tool_effects import ToolEffect
from app.tools.contracts import (
    CredentialSource,
    GraphToolPolicy,
    ToolExecutionContext,
    ToolInputModel,
    ToolOutputModel,
    ToolSpec,
)
from app.tools.registry import ToolRegistry

LOOKUP_SYNTHETIC_RECORD_TOOL_NAME = "lookup_synthetic_record"
MANUAL_LEARNING_POLICY_NAME = "manual_learning_loop"

_SYNTHETIC_RECORDS = {
    "record-1": "Synthetic backend role record.",
    "record-2": "Synthetic platform engineering record.",
}


class LookupSyntheticRecordInput(ToolInputModel):
    record_id: str = Field(max_length=64)

    @field_validator("record_id")
    @classmethod
    def record_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("record id must not be blank")
        return value


class LookupSyntheticRecordOutput(ToolOutputModel):
    record_id: str
    found: bool
    summary: str | None

    @model_validator(mode="after")
    def result_fields_must_agree(self) -> "LookupSyntheticRecordOutput":
        if self.found != (self.summary is not None):
            raise ValueError("found and summary must agree")
        return self


async def lookup_synthetic_record(
    tool_input: ToolInputModel,
    _context: ToolExecutionContext,
) -> LookupSyntheticRecordOutput:
    if not isinstance(tool_input, LookupSyntheticRecordInput):
        raise TypeError("lookup tool received the wrong input contract")
    return resolve_synthetic_record(tool_input)


def resolve_synthetic_record(
    tool_input: LookupSyntheticRecordInput,
) -> LookupSyntheticRecordOutput:
    """Resolve fixed teaching data without execution context or side effects."""
    summary = _SYNTHETIC_RECORDS.get(tool_input.record_id)
    return LookupSyntheticRecordOutput(
        record_id=tool_input.record_id,
        found=summary is not None,
        summary=summary,
    )


LOOKUP_SYNTHETIC_RECORD_SPEC = ToolSpec(
    name=LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
    description="Look up one deterministic synthetic learning record.",
    input_model=LookupSyntheticRecordInput,
    output_model=LookupSyntheticRecordOutput,
    effect=ToolEffect.READ_ONLY,
    credential_source=CredentialSource.NONE,
    timeout_seconds=1.0,
    max_attempts=1,
    per_run_call_limit=3,
    max_output_bytes=4096,
    handler=lookup_synthetic_record,
)

MANUAL_LEARNING_POLICY = GraphToolPolicy(
    name=MANUAL_LEARNING_POLICY_NAME,
    allowed_tool_names=frozenset({LOOKUP_SYNTHETIC_RECORD_TOOL_NAME}),
    allowed_effects=frozenset({ToolEffect.READ_ONLY}),
)


def create_teaching_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        specs=(LOOKUP_SYNTHETIC_RECORD_SPEC,),
        policies=(MANUAL_LEARNING_POLICY,),
    )
