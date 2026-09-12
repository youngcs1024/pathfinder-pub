from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import Field, StringConstraints

from app.domain.action_execution import ActionExecutionStore
from app.domain.actions import (
    SUBMIT_APPLICATION_TOOL_NAME,
    AnswerKey,
    AnswerText,
    CoverLetterText,
    JobReference,
    SubmitApplicationArgsV1,
    TrustedActionTargetV1,
)
from app.domain.tool_effects import ToolEffect
from app.tools.adapters.mock_portal import MockPortalHTTPAdapter
from app.tools.contracts import (
    CredentialSource,
    GraphToolPolicy,
    ToolExecutionContext,
    ToolInputModel,
    ToolOutputModel,
    ToolSpec,
)
from app.tools.registry import ApprovedActionFaultInjector, ToolRegistry

APPROVED_ACTION_POLICY_NAME = "approved_mock_action"


class SubmitMockApplicationInput(ToolInputModel):
    job_ref: JobReference
    resume_document_id: UUID
    answers: dict[AnswerKey, AnswerText] = Field(default_factory=dict, max_length=32)
    cover_letter: CoverLetterText


ExternalRef = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=200)]


class SubmitMockApplicationOutput(ToolOutputModel):
    external_ref: ExternalRef


class SubmitMockApplicationHandler:
    def __init__(self, adapter: MockPortalHTTPAdapter) -> None:
        self._adapter = adapter

    async def __call__(
        self,
        tool_input: ToolInputModel,
        context: ToolExecutionContext,
    ) -> object:
        if not isinstance(tool_input, SubmitMockApplicationInput):
            raise TypeError("mock application input uses the wrong contract")
        if (
            context.action_intent_id is None
            or context.approval_request_id is None
            or context.trusted_target is None
            or context.budget.call_limit != 1
            or context.budget.call_number != 1
        ):
            raise PermissionError("trusted approved action context is required")
        target = TrustedActionTargetV1.model_validate(dict(context.trusted_target), strict=True)
        if target != TrustedActionTargetV1():
            raise PermissionError("trusted mock target is invalid")
        payload = SubmitApplicationArgsV1.model_validate(
            tool_input.model_dump(mode="python", round_trip=True), strict=True
        )
        external_ref = await self._adapter.submit(
            workspace_id=context.workspace_id,
            actor_user_id=context.actor_user_id,
            run_id=context.run_id,
            action_intent_id=context.action_intent_id,
            idempotency_key=str(context.action_intent_id),
            payload=payload,
        )
        return SubmitMockApplicationOutput(external_ref=external_ref)


def create_submit_mock_application_spec(adapter: MockPortalHTTPAdapter) -> ToolSpec:
    return ToolSpec(
        name=SUBMIT_APPLICATION_TOOL_NAME,
        description="Submit an exactly approved application payload to the internal Mock Portal.",
        input_model=SubmitMockApplicationInput,
        output_model=SubmitMockApplicationOutput,
        effect=ToolEffect.IRREVERSIBLE,
        credential_source=CredentialSource.NONE,
        timeout_seconds=10.0,
        max_attempts=1,
        per_run_call_limit=1,
        max_output_bytes=1_000,
        handler=SubmitMockApplicationHandler(adapter),
    )


def create_approved_action_registry(
    *,
    adapter: MockPortalHTTPAdapter,
    action_execution_store: ActionExecutionStore,
    fault_injector: ApprovedActionFaultInjector | None = None,
    action_recovery_max_attempts: int = 3,
) -> ToolRegistry:
    spec = create_submit_mock_application_spec(adapter)
    return ToolRegistry(
        specs=(spec,),
        policies=(
            GraphToolPolicy(
                name=APPROVED_ACTION_POLICY_NAME,
                allowed_tool_names=frozenset({spec.name}),
                allowed_effects=frozenset({ToolEffect.IRREVERSIBLE}),
            ),
        ),
        action_execution_store=action_execution_store,
        approved_action_adapter=adapter,
        action_recovery_max_attempts=action_recovery_max_attempts,
        approved_action_fault_injector=fault_injector,
    )
