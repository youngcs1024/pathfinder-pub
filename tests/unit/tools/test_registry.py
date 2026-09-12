import asyncio
import json
from dataclasses import replace
from enum import StrEnum
from typing import cast
from uuid import UUID

import pytest
from pydantic import ConfigDict, Field, ValidationError

from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import ToolInvocationReservation
from app.llm.ports import ModelToolCall
from app.tools.contracts import (
    CancellationCheck,
    CredentialSource,
    GraphToolPolicy,
    ToolExecutionContext,
    ToolInputModel,
    ToolOutputModel,
    ToolRunContext,
    ToolSpec,
    ToolTransientError,
)
from app.tools.registry import (
    ToolCallLimitExceededError,
    ToolCancelledError,
    ToolConfigurationError,
    ToolDeadlineExceededError,
    ToolExecutionError,
    ToolInputValidationError,
    ToolNotAllowedError,
    ToolOutputTooLargeError,
    ToolOutputValidationError,
    ToolRegistry,
    ToolTimeoutError,
    ToolUnavailableError,
)

WORKSPACE_ID = UUID("00000000-0000-4000-8000-000000000001")
ACTOR_USER_ID = UUID("00000000-0000-4000-8000-000000000002")
RUN_ID = UUID("00000000-0000-4000-8000-000000000003")
ACTION_INTENT_ID = UUID("00000000-0000-4000-8000-000000000004")
APPROVAL_REQUEST_ID = UUID("00000000-0000-4000-8000-000000000005")


class _Cancellation:
    def __init__(self, *, cancelled: bool = False) -> None:
        self.cancelled = cancelled

    def is_cancelled(self) -> bool:
        return self.cancelled


class _RecordInput(ToolInputModel):
    record_id: str = Field(min_length=1, max_length=64)


class _RecordOutput(ToolOutputModel):
    record_id: str
    result: str


class _RecordingHandler:
    def __init__(self, actions: list[object] | None = None) -> None:
        self.actions = list(actions or [])
        self.inputs: list[ToolInputModel] = []
        self.contexts: list[ToolExecutionContext] = []

    async def __call__(
        self,
        tool_input: ToolInputModel,
        context: ToolExecutionContext,
    ) -> object:
        self.inputs.append(tool_input)
        self.contexts.append(context)
        action = (
            self.actions.pop(0)
            if self.actions
            else _RecordOutput(record_id="record-default", result="synthetic")
        )
        if isinstance(action, BaseException):
            raise action
        return action


class _FailureRecorder:
    def __init__(self) -> None:
        self.error_categories: list[str] = []

    async def reserve(self, *, invocation_id: UUID, **_kwargs: object):
        return ToolInvocationReservation(invocation_id=invocation_id, call_number=1)

    async def start_attempt(self, **_kwargs: object) -> None:
        return None

    async def succeed(self, **_kwargs: object) -> None:
        return None

    async def fail(self, **kwargs: object) -> None:
        self.error_categories.append(cast(str, kwargs["error_category"]))


class _InvocationIds:
    def __init__(self, start: int = 100) -> None:
        self._next_value = start

    def __call__(self) -> UUID:
        value = UUID(int=self._next_value)
        self._next_value += 1
        return value


def _spec(
    handler: object,
    *,
    name: str = "lookup_record",
    input_model: type[ToolInputModel] = _RecordInput,
    output_model: type[ToolOutputModel] = _RecordOutput,
    effect: ToolEffect = ToolEffect.READ_ONLY,
    credential_source: CredentialSource = CredentialSource.NONE,
    timeout_seconds: float = 1.0,
    max_attempts: int = 1,
    per_run_call_limit: int = 3,
    max_output_bytes: int = 4096,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Look up a synthetic record.",
        input_model=input_model,
        output_model=output_model,
        effect=effect,
        credential_source=credential_source,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        per_run_call_limit=per_run_call_limit,
        max_output_bytes=max_output_bytes,
        handler=handler,  # type: ignore[arg-type]
    )


def _policy(
    *names: str,
    allowed_effects: frozenset[ToolEffect] = frozenset({ToolEffect.READ_ONLY}),
    name: str = "manual_learning_loop",
) -> GraphToolPolicy:
    return GraphToolPolicy(
        name=name,
        allowed_tool_names=frozenset(names or ("lookup_record",)),
        allowed_effects=allowed_effects,
    )


def _registry(
    spec: ToolSpec,
    *,
    policy: GraphToolPolicy | None = None,
    recorder: object | None = None,
) -> ToolRegistry:
    return ToolRegistry(
        specs=(spec,),
        policies=(policy or _policy(spec.name),),
        recorder=recorder,  # type: ignore[arg-type]
    )


def _run_context(
    *,
    cancellation: CancellationCheck | None = None,
    deadline: float = 100.0,
    trusted_target: dict[str, object] | None = None,
) -> ToolRunContext:
    return ToolRunContext(
        workspace_id=WORKSPACE_ID,
        actor_user_id=ACTOR_USER_ID,
        run_id=RUN_ID,
        action_intent_id=ACTION_INTENT_ID,
        approval_request_id=APPROVAL_REQUEST_ID,
        trusted_target=trusted_target,  # type: ignore[arg-type]
        deadline=deadline,
        cancellation=cancellation or _Cancellation(),
    )


def _bind(
    registry: ToolRegistry,
    *,
    context: ToolRunContext | None = None,
    policy_name: str = "manual_learning_loop",
    clock_value: float = 10.0,
):
    return registry.bind(
        policy_name=policy_name,
        context=context or _run_context(),
        invocation_id_factory=_InvocationIds(),
        clock=lambda: clock_value,
    )


def _call(
    *,
    name: str = "lookup_record",
    call_id: str = "call-1",
    arguments: dict[str, object] | None = None,
) -> ModelToolCall:
    return ModelToolCall(
        call_id=call_id,
        name=name,
        arguments=({"record_id": "record-1"} if arguments is None else arguments),  # type: ignore[arg-type]
    )


def test_tool_effect_and_credential_source_use_locked_values() -> None:
    assert [effect.value for effect in ToolEffect] == [
        "read_only",
        "reversible",
        "irreversible",
    ]
    assert [source.value for source in CredentialSource] == ["none", "server_managed"]
    assert issubclass(ToolEffect, StrEnum)


def test_tool_contract_models_are_strict_frozen_and_hide_input() -> None:
    value = _RecordInput(record_id="record-1")

    with pytest.raises(ValidationError, match="frozen"):
        value.record_id = "changed"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _RecordInput.model_validate({"record_id": "record-1", "workspace_id": "forged"})
    with pytest.raises(ValidationError):
        _RecordInput.model_validate({"record_id": 1})


@pytest.mark.parametrize(
    ("invalid_spec", "message"),
    [
        (_spec(_RecordingHandler(), name="Bad-Name"), "lower snake case"),
        (_spec(_RecordingHandler(), name="_bad_name"), "lower snake case"),
        (_spec(_RecordingHandler(), name="bad_name_"), "lower snake case"),
        (_spec(_RecordingHandler(), name="bad__name"), "lower snake case"),
        (replace(_spec(_RecordingHandler()), description=" "), "must not be blank"),
        (replace(_spec(_RecordingHandler()), timeout_seconds=0), "positive finite"),
        (replace(_spec(_RecordingHandler()), timeout_seconds=float("inf")), "positive finite"),
        (replace(_spec(_RecordingHandler()), max_attempts=True), "positive integer"),
        (replace(_spec(_RecordingHandler()), per_run_call_limit=0), "positive integer"),
        (replace(_spec(_RecordingHandler()), max_output_bytes=-1), "positive integer"),
        (
            replace(
                _spec(_RecordingHandler()),
                effect=cast(ToolEffect, "read_only"),
            ),
            "authoritative enum",
        ),
        (
            replace(
                _spec(_RecordingHandler()),
                credential_source=cast(CredentialSource, "none"),
            ),
            "credential source",
        ),
    ],
)
def test_registry_rejects_invalid_tool_configuration(
    invalid_spec: ToolSpec,
    message: str,
) -> None:
    with pytest.raises(ToolConfigurationError, match=message):
        ToolRegistry(specs=(invalid_spec,), policies=(_policy(invalid_spec.name),))


def test_registry_rejects_sync_handler_and_weakened_contract_model() -> None:
    def sync_handler(_tool_input: ToolInputModel, _context: ToolExecutionContext) -> object:
        return {}

    class WeakInput(ToolInputModel):
        model_config = ConfigDict(extra="allow")
        record_id: str

    with pytest.raises(ToolConfigurationError, match="asynchronous"):
        _registry(_spec(sync_handler))
    with pytest.raises(ToolConfigurationError, match="weakens"):
        _registry(_spec(_RecordingHandler(), input_model=WeakInput))


def test_registry_rejects_input_contract_without_model_visible_json_schema() -> None:
    class OpaqueValue:
        pass

    class OpaqueInput(ToolInputModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        opaque: OpaqueValue

    with pytest.raises(ToolConfigurationError, match="schema generation failed"):
        _registry(_spec(_RecordingHandler(), input_model=OpaqueInput))


@pytest.mark.parametrize(
    "field_name",
    [
        "workspace_id",
        "actor_user_id",
        "run_id",
        "invocation_id",
        "action_intent_id",
        "approval_request_id",
        "target",
        "trusted_target",
        "credential",
        "credentials",
        "credential_source",
        "deadline",
        "budget",
        "cancellation",
    ],
)
def test_registry_rejects_contracts_that_declare_reserved_fields(field_name: str) -> None:
    ReservedInput = type(
        "ReservedInput",
        (ToolInputModel,),
        {
            "__annotations__": {field_name: str},
            field_name: "",
        },
    )

    with pytest.raises(ToolConfigurationError, match="reserved execution field"):
        _registry(_spec(_RecordingHandler(), input_model=ReservedInput))


def test_registry_rejects_reserved_input_aliases_visible_to_the_model() -> None:
    class AliasedReservedInput(ToolInputModel):
        business_value: str = Field(alias="workspace_id")

    with pytest.raises(ToolConfigurationError, match="reserved execution field"):
        _registry(_spec(_RecordingHandler(), input_model=AliasedReservedInput))


def test_registry_rejects_duplicate_and_incomplete_configuration() -> None:
    spec = _spec(_RecordingHandler())
    policy = _policy(spec.name)

    with pytest.raises(ToolConfigurationError, match="at least one tool"):
        ToolRegistry(specs=(), policies=(policy,))
    with pytest.raises(ToolConfigurationError, match="unique"):
        ToolRegistry(specs=(spec, spec), policies=(policy,))
    with pytest.raises(ToolConfigurationError, match="at least one graph policy"):
        ToolRegistry(specs=(spec,), policies=())
    with pytest.raises(ToolConfigurationError, match="unique"):
        ToolRegistry(specs=(spec,), policies=(policy, policy))


def test_registry_rejects_unknown_policy_tool_and_effect_mismatch() -> None:
    spec = _spec(_RecordingHandler())

    with pytest.raises(ToolConfigurationError, match="unregistered"):
        ToolRegistry(specs=(spec,), policies=(_policy("unknown_tool"),))
    with pytest.raises(ToolConfigurationError, match="does not authorize"):
        ToolRegistry(
            specs=(spec,),
            policies=(_policy(spec.name, allowed_effects=frozenset({ToolEffect.REVERSIBLE})),),
        )


def test_registry_rejects_retry_for_non_read_only_effect() -> None:
    for effect in (ToolEffect.REVERSIBLE, ToolEffect.IRREVERSIBLE):
        with pytest.raises(ToolConfigurationError, match="only read-only"):
            _registry(_spec(_RecordingHandler(), effect=effect, max_attempts=2))


def test_bind_rejects_non_json_trusted_target_with_safe_error() -> None:
    canary = "trusted-target-private-value"

    class OpaqueValue:
        def __repr__(self) -> str:
            return canary

    context = _run_context(trusted_target={"opaque": OpaqueValue()})  # type: ignore[dict-item]

    with pytest.raises(ToolConfigurationError, match="must contain JSON values") as captured:
        _registry(_spec(_RecordingHandler())).bind(
            policy_name="manual_learning_loop",
            context=context,
        )

    assert canary not in str(captured.value)
    assert canary not in repr(captured.value)


def test_model_schemas_are_derived_filtered_sorted_and_defensively_copied() -> None:
    alpha = _spec(_RecordingHandler(), name="alpha_tool")
    zeta = _spec(_RecordingHandler(), name="zeta_tool")
    registry = ToolRegistry(
        specs=(zeta, alpha),
        policies=(_policy("zeta_tool", "alpha_tool"),),
    )
    runtime = _bind(registry)

    first = runtime.model_tools()
    first[0].input_schema["tampered"] = True
    second = runtime.model_tools()
    serialized = json.dumps([tool.model_dump(mode="json") for tool in second])

    assert [tool.name for tool in second] == ["alpha_tool", "zeta_tool"]
    assert second[0].input_schema["additionalProperties"] is False
    assert set(second[0].input_schema["properties"]) == {"record_id"}
    assert "tampered" not in second[0].input_schema
    for forbidden in (
        "workspace_id",
        "actor_user_id",
        "trusted_target",
        "credential_source",
        "deadline",
        "budget",
        "cancellation",
        "handler",
        "timeout_seconds",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_runtime_validates_input_passes_trusted_context_and_normalizes_output() -> None:
    first_invocation = UUID(int=100)
    handler = _RecordingHandler([_RecordOutput(record_id="record-1", result="synthetic")])
    runtime = _bind(_registry(_spec(handler)))

    result = await runtime.execute(_call())

    assert result == '{"record_id":"record-1","result":"synthetic"}'
    assert handler.inputs == [_RecordInput(record_id="record-1")]
    assert len(handler.contexts) == 1
    context = handler.contexts[0]
    assert context.workspace_id == WORKSPACE_ID
    assert context.actor_user_id == ACTOR_USER_ID
    assert context.run_id == RUN_ID
    assert context.invocation_id == first_invocation
    assert context.action_intent_id == ACTION_INTENT_ID
    assert context.approval_request_id == APPROVAL_REQUEST_ID
    assert context.deadline == 100.0
    assert context.budget.call_number == 1
    assert context.budget.call_limit == 3
    assert context.budget.remaining_calls == 2
    assert isinstance(context.cancellation, CancellationCheck)


@pytest.mark.asyncio
async def test_sequential_logical_calls_receive_distinct_invocation_ids_and_budgets() -> None:
    handler = _RecordingHandler(
        [
            _RecordOutput(record_id="record-1", result="first"),
            _RecordOutput(record_id="record-2", result="second"),
        ]
    )
    runtime = _bind(_registry(_spec(handler)))

    await runtime.execute(_call(call_id="call-1", arguments={"record_id": "record-1"}))
    await runtime.execute(_call(call_id="call-2", arguments={"record_id": "record-2"}))

    assert [context.invocation_id for context in handler.contexts] == [UUID(int=100), UUID(int=101)]
    assert [context.budget.call_number for context in handler.contexts] == [1, 2]
    assert [context.budget.remaining_calls for context in handler.contexts] == [2, 1]


@pytest.mark.asyncio
async def test_validate_call_is_pure_and_execute_still_revalidates() -> None:
    handler = _RecordingHandler([_RecordOutput(record_id="record-1", result="synthetic")])
    allocated_ids: list[UUID] = []

    def invocation_id_factory() -> UUID:
        value = UUID(int=100 + len(allocated_ids))
        allocated_ids.append(value)
        return value

    runtime = _registry(_spec(handler)).bind(
        policy_name="manual_learning_loop",
        context=_run_context(),
        invocation_id_factory=invocation_id_factory,
        clock=lambda: 10.0,
    )
    call = _call()

    runtime.validate_call(call)

    assert allocated_ids == []
    assert handler.inputs == []
    assert handler.contexts == []

    assert await runtime.execute(call) == '{"record_id":"record-1","result":"synthetic"}'
    assert allocated_ids == [UUID(int=100)]
    assert handler.contexts[0].budget.call_number == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        _call(name="unknown_tool"),
        _call(arguments={}),
        _call(arguments={"record_id": "record-1", "extra": "forged"}),
        _call(arguments={"record_id": "record-1", "workspace_id": "forged"}),
    ],
    ids=["unknown", "missing", "extra", "reserved"],
)
async def test_validate_call_and_execute_share_deterministic_contract(
    call: ModelToolCall,
) -> None:
    handler = _RecordingHandler()
    runtime = _bind(_registry(_spec(handler)))

    with pytest.raises((ToolNotAllowedError, ToolInputValidationError)) as validation_error:
        runtime.validate_call(call)
    with pytest.raises(type(validation_error.value)) as execution_error:
        await runtime.execute(call)

    assert str(execution_error.value) == str(validation_error.value)
    assert handler.inputs == []
    assert handler.contexts == []


@pytest.mark.asyncio
async def test_unknown_and_registered_but_disallowed_tools_fail_before_handler() -> None:
    allowed_handler = _RecordingHandler()
    blocked_handler = _RecordingHandler()
    allowed = _spec(allowed_handler, name="allowed_tool")
    blocked = _spec(blocked_handler, name="blocked_tool")
    registry = ToolRegistry(
        specs=(allowed, blocked),
        policies=(_policy("allowed_tool"),),
    )
    runtime = _bind(registry)

    for name in ("unknown_tool", "blocked_tool"):
        with pytest.raises(ToolNotAllowedError, match="not allowed"):
            await runtime.execute(_call(name=name))

    assert allowed_handler.inputs == []
    assert blocked_handler.inputs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"record_id": 1},
        {"record_id": "record-1", "extra": "forged"},
    ],
    ids=["missing", "wrong-type", "extra"],
)
async def test_invalid_business_arguments_fail_before_handler(
    arguments: dict[str, object],
) -> None:
    handler = _RecordingHandler()
    runtime = _bind(_registry(_spec(handler)))

    with pytest.raises(ToolInputValidationError, match="failed validation"):
        await runtime.execute(_call(arguments=arguments))

    assert handler.inputs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field_name",
    [
        "workspace_id",
        "actor_user_id",
        "run_id",
        "invocation_id",
        "action_intent_id",
        "approval_request_id",
        "target",
        "trusted_target",
        "credential",
        "credentials",
        "credential_source",
        "deadline",
        "budget",
        "cancellation",
    ],
)
async def test_model_cannot_supply_reserved_execution_fields(field_name: str) -> None:
    handler = _RecordingHandler()
    runtime = _bind(_registry(_spec(handler)))
    arguments = {"record_id": "record-1", field_name: "forged"}

    with pytest.raises(ToolInputValidationError, match="reserved execution field"):
        await runtime.execute(_call(arguments=arguments))

    assert handler.inputs == []


@pytest.mark.asyncio
async def test_call_limit_is_per_bound_runtime() -> None:
    handler = _RecordingHandler(
        [
            _RecordOutput(record_id="record-1", result="one"),
            _RecordOutput(record_id="record-1", result="two"),
        ]
    )
    registry = _registry(_spec(handler, per_run_call_limit=1))
    first_runtime = _bind(registry)

    await first_runtime.execute(_call(call_id="first"))
    with pytest.raises(ToolCallLimitExceededError, match="limit"):
        await first_runtime.execute(_call(call_id="blocked"))

    second_runtime = _bind(registry)
    assert await second_runtime.execute(_call(call_id="second")) == (
        '{"record_id":"record-1","result":"two"}'
    )
    assert len(handler.inputs) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_result", "expected_error"),
    [
        (RuntimeError("private failure"), ToolExecutionError),
        (TimeoutError("private timeout"), ToolTimeoutError),
        ({"record_id": "record-1"}, ToolOutputValidationError),
    ],
    ids=["permanent-error", "timeout", "invalid-output"],
)
async def test_failed_started_call_consumes_one_logical_call(
    handler_result: object,
    expected_error: type[Exception],
) -> None:
    handler = _RecordingHandler([handler_result])
    runtime = _bind(_registry(_spec(handler, per_run_call_limit=1)))

    with pytest.raises(expected_error):
        await runtime.execute(_call(call_id="failed"))
    with pytest.raises(ToolCallLimitExceededError, match="limit"):
        await runtime.execute(_call(call_id="blocked"))

    assert len(handler.inputs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_failure",
    [TimeoutError("private timeout"), ToolTransientError()],
    ids=["timeout", "typed-transient"],
)
async def test_read_only_retry_reuses_input_context_and_invocation(
    first_failure: Exception,
) -> None:
    handler = _RecordingHandler(
        [
            first_failure,
            _RecordOutput(record_id="record-1", result="recovered"),
        ]
    )
    runtime = _bind(_registry(_spec(handler, max_attempts=2, per_run_call_limit=1)))

    result = await runtime.execute(_call())

    assert result == '{"record_id":"record-1","result":"recovered"}'
    assert len(handler.inputs) == 2
    assert handler.inputs[0] is handler.inputs[1]
    assert handler.contexts[0] is handler.contexts[1]
    assert handler.contexts[0].invocation_id == UUID(int=100)
    assert handler.contexts[0].budget.call_number == 1


@pytest.mark.asyncio
async def test_exhausted_timeout_and_transient_attempts_have_distinct_safe_errors() -> None:
    timeout_handler = _RecordingHandler([TimeoutError("secret"), TimeoutError("secret")])
    timeout_runtime = _bind(_registry(_spec(timeout_handler, max_attempts=2)))

    with pytest.raises(ToolTimeoutError, match="timeout attempts") as timeout_error:
        await timeout_runtime.execute(_call())
    assert "secret" not in repr(timeout_error.value)
    assert len(timeout_handler.inputs) == 2

    transient_handler = _RecordingHandler([ToolTransientError(), ToolTransientError()])
    recorder = _FailureRecorder()
    transient_runtime = _bind(
        _registry(_spec(transient_handler, max_attempts=2), recorder=recorder)
    )
    with pytest.raises(ToolUnavailableError, match="transient attempts"):
        await transient_runtime.execute(_call())
    assert len(transient_handler.inputs) == 2
    assert recorder.error_categories == ["provider_unavailable"]


@pytest.mark.asyncio
async def test_permanent_handler_error_is_not_retried_or_reflected() -> None:
    canary = "private-handler-error-canary"
    handler = _RecordingHandler([RuntimeError(canary)])
    runtime = _bind(_registry(_spec(handler, max_attempts=3)))

    with pytest.raises(ToolExecutionError, match=r"^tool handler failed$") as captured:
        await runtime.execute(_call())

    assert len(handler.inputs) == 1
    assert canary not in str(captured.value)
    assert canary not in repr(captured.value)


@pytest.mark.asyncio
async def test_async_timeout_wrapper_stops_a_hanging_handler() -> None:
    calls: list[ToolExecutionContext] = []

    async def hanging_handler(
        _tool_input: ToolInputModel,
        context: ToolExecutionContext,
    ) -> object:
        calls.append(context)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    runtime = _registry(_spec(hanging_handler, timeout_seconds=0.01)).bind(
        policy_name="manual_learning_loop",
        context=_run_context(deadline=1000.0),
        invocation_id_factory=_InvocationIds(),
        clock=lambda: 10.0,
    )

    with pytest.raises(ToolTimeoutError, match="timeout attempts"):
        await runtime.execute(_call())

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_deadline_that_expires_during_handler_is_not_retried_as_tool_timeout() -> None:
    calls: list[ToolExecutionContext] = []

    async def hanging_handler(
        _tool_input: ToolInputModel,
        context: ToolExecutionContext,
    ) -> object:
        calls.append(context)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    runtime = _registry(_spec(hanging_handler, timeout_seconds=1.0, max_attempts=2)).bind(
        policy_name="manual_learning_loop",
        context=_run_context(deadline=10.01),
        invocation_id_factory=_InvocationIds(),
        clock=lambda: 10.0,
    )

    with pytest.raises(ToolDeadlineExceededError, match="expired during an attempt"):
        await runtime.execute(_call())

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_cancellation_and_expired_deadline_stop_before_handler_and_budget_reservation() -> (
    None
):
    cancellation = _Cancellation(cancelled=True)
    handler = _RecordingHandler([_RecordOutput(record_id="record-1", result="allowed")])
    runtime = _bind(
        _registry(_spec(handler, per_run_call_limit=1)),
        context=_run_context(cancellation=cancellation),
    )

    with pytest.raises(ToolCancelledError, match="cancelled"):
        await runtime.execute(_call(call_id="cancelled"))
    cancellation.cancelled = False
    assert await runtime.execute(_call(call_id="allowed")) == (
        '{"record_id":"record-1","result":"allowed"}'
    )
    assert handler.contexts[0].budget.call_number == 1

    expired_handler = _RecordingHandler()
    expired_runtime = _bind(
        _registry(_spec(expired_handler)),
        context=_run_context(deadline=10.0),
        clock_value=10.0,
    )
    with pytest.raises(ToolDeadlineExceededError, match="expired"):
        await expired_runtime.execute(_call())
    assert expired_handler.inputs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_output",
    [
        {"record_id": "record-1"},
        {"record_id": "record-1", "result": "ok", "extra": "forged"},
        object(),
    ],
    ids=["missing", "extra", "wrong-type"],
)
async def test_invalid_handler_output_fails_without_model_visible_result(
    invalid_output: object,
) -> None:
    handler = _RecordingHandler([invalid_output])
    runtime = _bind(_registry(_spec(handler)))

    with pytest.raises(ToolOutputValidationError, match="output failed validation"):
        await runtime.execute(_call())


@pytest.mark.asyncio
async def test_non_json_and_oversized_outputs_fail_safely() -> None:
    class OpaqueOutput(ToolOutputModel):
        payload: object

    opaque_handler = _RecordingHandler([{"payload": object()}])
    opaque_runtime = _bind(_registry(_spec(opaque_handler, output_model=OpaqueOutput)))
    with pytest.raises(ToolOutputValidationError, match="output failed validation"):
        await opaque_runtime.execute(_call())

    large_handler = _RecordingHandler([_RecordOutput(record_id="record-1", result="é" * 100)])
    large_runtime = _bind(_registry(_spec(large_handler, max_output_bytes=32)))
    with pytest.raises(ToolOutputTooLargeError, match="byte limit"):
        await large_runtime.execute(_call())


@pytest.mark.asyncio
async def test_context_and_validation_canaries_do_not_reach_observable_surfaces(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "trusted-context-secret-canary"
    handler = _RecordingHandler([RuntimeError(canary)])
    runtime = _bind(
        _registry(_spec(handler)),
        context=_run_context(trusted_target={"opaque": canary}),
    )

    with pytest.raises(ToolExecutionError) as execution_error:
        await runtime.execute(_call())
    with pytest.raises(ToolInputValidationError) as validation_error:
        await runtime.execute(_call(arguments={"record_id": canary, "extra": canary}))

    captured = capsys.readouterr()
    assert handler.contexts[0].trusted_target == {"opaque": canary}
    observable = "\n".join(
        (
            json.dumps([tool.model_dump(mode="json") for tool in runtime.model_tools()]),
            repr(handler.contexts[0]),
            str(execution_error.value),
            repr(execution_error.value),
            str(validation_error.value),
            repr(validation_error.value),
            caplog.text,
            captured.out,
            captured.err,
        )
    )
    assert canary not in observable


@pytest.mark.asyncio
async def test_handler_cannot_mutate_original_nested_model_arguments() -> None:
    class NestedInput(ToolInputModel):
        payload: dict[str, str]

    class NestedOutput(ToolOutputModel):
        value: str

    async def mutating_handler(
        tool_input: ToolInputModel,
        _context: ToolExecutionContext,
    ) -> object:
        assert isinstance(tool_input, NestedInput)
        tool_input.payload["value"] = "changed"
        return NestedOutput(value=tool_input.payload["value"])

    runtime = _bind(
        _registry(
            _spec(
                mutating_handler,
                input_model=NestedInput,
                output_model=NestedOutput,
            )
        )
    )
    arguments = {"payload": {"value": "original"}}
    call = _call(arguments=arguments)

    assert await runtime.execute(call) == '{"value":"changed"}'
    assert call.arguments == {"payload": {"value": "original"}}
    assert arguments == {"payload": {"value": "original"}}


@pytest.mark.parametrize("retry_error", [None, TimeoutError(), ToolTransientError()])
async def test_trace_one_invocation_aggregates_attempts_and_excludes_bodies(retry_error) -> None:
    from app.domain.tracing import current_trace_scope
    from tests.tracing import CollectingTraceSink, collecting_node

    output = _RecordOutput(record_id="OUTPUT-CANARY", result="完成🧭")
    handler = _RecordingHandler(([retry_error] if retry_error else []) + [output])
    recorder = _FailureRecorder()
    runtime = _bind(
        _registry(_spec(handler, max_attempts=2), recorder=recorder),
        context=_run_context(trusted_target={"target": "TRUSTED-TARGET-CANARY"}),
    )
    sink = CollectingTraceSink()
    with collecting_node(sink) as scope:
        result = await runtime.execute(_call(arguments={"record_id": "ARGUMENTS-CANARY"}))
        assert current_trace_scope() == scope
    [(context, start)] = [v for v in sink.starts.values() if v[1].span_kind == "tool"]
    assert start.parent == scope.parent
    assert dict(start.metadata) == {
        "tool_name": "lookup_record",
        "tool_effect": "read_only",
        "tool_invocation_id": handler.contexts[0].invocation_id,
    }
    assert {ctx.invocation_id for ctx in handler.contexts} == {UUID(int=100)}
    finish = sink.finishes[context.context_id]
    attempts = 2 if retry_error else 1
    assert len(handler.inputs) == attempts
    assert finish.status == "succeeded"
    assert dict(finish.metadata) == {
        "attempt_number": attempts,
        "retry_count": attempts - 1,
        "output_bytes": len(result.encode("utf-8")),
    }
    for canary in ("ARGUMENTS-CANARY", "OUTPUT-CANARY", "TRUSTED-TARGET-CANARY"):
        assert canary not in sink.safe_json() + repr(sink.starts) + repr(sink.finishes)


@pytest.mark.parametrize(
    "action,exception,category,attempts",
    [
        (TimeoutError("EXCEPTION-CANARY"), ToolTimeoutError, "provider_timeout", 2),
        (ToolTransientError(), ToolUnavailableError, "provider_unavailable", 2),
        (RuntimeError("EXCEPTION-CANARY"), ToolExecutionError, "tool_execution_failed", 1),
        ({"bad": "OUTPUT-CANARY"}, ToolOutputValidationError, "invalid_tool_output", 1),
        (asyncio.CancelledError(), asyncio.CancelledError, "cancelled", 1),
    ],
)
async def test_trace_terminal_failure_preserves_durable_accounting(
    action, exception, category, attempts
) -> None:
    from app.domain.tracing import current_trace_scope
    from tests.tracing import CollectingTraceSink, collecting_node

    recorder = _FailureRecorder()
    handler = _RecordingHandler([action, action])
    runtime = _bind(_registry(_spec(handler, max_attempts=2), recorder=recorder))
    sink = CollectingTraceSink()
    with collecting_node(sink) as scope:
        with pytest.raises(exception):
            await runtime.execute(_call())
        assert current_trace_scope() == scope
        assert recorder.error_categories == [category]
    [finish] = [v for v in sink.finishes.values() if v.span_kind == "tool"]
    assert finish.status == ("cancelled" if category == "cancelled" else "failed")
    assert finish.error_category == category
    assert dict(finish.metadata) == {"attempt_number": attempts, "retry_count": attempts - 1}
    assert "EXCEPTION-CANARY" not in sink.safe_json() + repr(sink.finishes)
    assert "OUTPUT-CANARY" not in sink.safe_json() + repr(sink.starts)


@pytest.mark.parametrize(
    "rejection", ["invalid", "reserved", "unknown", "disallowed", "limit", "cancel", "deadline"]
)
async def test_trace_never_invents_pre_reservation_invocation(rejection) -> None:
    from tests.tracing import CollectingTraceSink, collecting_node

    handler = _RecordingHandler()
    context = _run_context(
        cancellation=_Cancellation(cancelled=rejection == "cancel"),
        deadline=0.0 if rejection == "deadline" else 100.0,
    )
    registry = _registry(_spec(handler, per_run_call_limit=1))
    if rejection == "disallowed":
        registry = ToolRegistry(
            specs=(_spec(handler), _spec(handler, name="not_allowed")), policies=(_policy(),)
        )
    runtime = _bind(registry, context=context)
    call = _call()
    exception = ToolInputValidationError
    if rejection == "invalid":
        call = _call(arguments={"record_id": 1})
    elif rejection == "reserved":
        call = _call(arguments={"record_id": "ok", "workspace_id": str(WORKSPACE_ID)})
    elif rejection in {"unknown", "disallowed"}:
        call = _call(name="not_allowed")
        exception = ToolNotAllowedError
    elif rejection == "limit":
        await runtime.execute(call)
        exception = ToolCallLimitExceededError
    elif rejection == "cancel":
        exception = ToolCancelledError
    elif rejection == "deadline":
        exception = ToolDeadlineExceededError
    sink = CollectingTraceSink()
    with collecting_node(sink), pytest.raises(exception):
        await runtime.execute(call)
    assert not [v for v in sink.starts.values() if v[1].span_kind == "tool"]


@pytest.mark.parametrize("mode", ["cancel", "deadline", "accounting"])
async def test_trace_no_attempt_until_recorder_start_returns(mode) -> None:
    from tests.tracing import CollectingTraceSink, collecting_node

    cancellation = _Cancellation()
    time_value = 10.0

    class Recorder(_FailureRecorder):
        async def reserve(self, **kwargs):
            nonlocal time_value
            result = await super().reserve(**kwargs)
            cancellation.cancelled = mode == "cancel"
            if mode == "deadline":
                time_value = 200.0
            return result

        async def start_attempt(self, **kwargs):
            raise RuntimeError("DB-EXCEPTION-CANARY")

    recorder = Recorder()
    handler = _RecordingHandler()
    runtime = _registry(_spec(handler), recorder=recorder).bind(
        policy_name="manual_learning_loop",
        context=_run_context(cancellation=cancellation),
        clock=lambda: time_value,
    )
    sink = CollectingTraceSink()
    with (
        collecting_node(sink),
        pytest.raises((ToolCancelledError, ToolDeadlineExceededError, ToolExecutionError)),
    ):
        await runtime.execute(_call())
    [finish] = [v for v in sink.finishes.values() if v.span_kind == "tool"]
    assert "attempt_number" not in finish.metadata
    assert handler.inputs == []
    assert "DB-EXCEPTION-CANARY" not in sink.safe_json()


@pytest.mark.parametrize("failure", ["start", "finish"])
async def test_tool_exporter_failure_cannot_change_business_result(failure) -> None:
    from tests.tracing import CollectingTraceSink, collecting_node

    class Sink(CollectingTraceSink):
        def start(self, span):
            if span.span_kind == "tool" and failure == "start":
                raise RuntimeError("export failure")
            return super().start(span)

        def finish(self, context, outcome):
            if context.span_kind == "tool" and failure == "finish":
                raise RuntimeError("export failure")
            super().finish(context, outcome)

    runtime = _bind(_registry(_spec(_RecordingHandler())))
    with collecting_node(Sink()):
        assert json.loads(await runtime.execute(_call()))["result"] == "synthetic"


@pytest.mark.parametrize("finish_failure", [False, True])
async def test_trace_task_cancellation_finishes_only_after_durable_failure(finish_failure) -> None:
    from app.domain.tracing import current_trace_scope
    from tests.tracing import CollectingTraceSink, collecting_node

    entered = asyncio.Event()
    accounting_entered = asyncio.Event()
    release_accounting = asyncio.Event()
    restored = []

    class Sink(CollectingTraceSink):
        def finish(self, context, outcome):
            if context.span_kind == "tool":
                assert recorder.error_categories == ["cancelled"]
                if finish_failure:
                    raise RuntimeError("export unavailable")
            super().finish(context, outcome)

    class Recorder(_FailureRecorder):
        async def fail(self, **kwargs):
            accounting_entered.set()
            await release_accounting.wait()
            await super().fail(**kwargs)

    async def handler(tool_input, context):
        entered.set()
        await asyncio.Event().wait()

    recorder = Recorder()
    sink = Sink()
    runtime = _bind(_registry(_spec(handler), recorder=recorder))
    with collecting_node(sink) as scope:

        async def execute():
            try:
                await runtime.execute(_call())
            finally:
                restored.append(current_trace_scope())

        task = asyncio.create_task(execute())
        await entered.wait()
        task.cancel()
        await accounting_entered.wait()
        assert not [f for f in sink.finishes.values() if f.span_kind == "tool"]
        release_accounting.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert restored == [scope]
    assert recorder.error_categories == ["cancelled"]
    if not finish_failure:
        [finish] = [f for f in sink.finishes.values() if f.span_kind == "tool"]
        assert finish.status == "cancelled"


@pytest.mark.parametrize("phase", ["reserve", "start_attempt"])
@pytest.mark.parametrize("revoked", [False, True])
async def test_recorder_unavailability_is_not_revocation_or_tool_retry(phase, revoked):
    from app.domain.errors import DomainUnavailableError
    from app.domain.tool_invocations import ToolInvocationAuthorizationError

    calls = []

    class Recorder(_FailureRecorder):
        async def reserve(self, **kwargs):
            calls.append("reserve")
            if phase == "reserve":
                raise ToolInvocationAuthorizationError() if revoked else DomainUnavailableError()
            return await super().reserve(**kwargs)

        async def start_attempt(self, **kwargs):
            calls.append("start_attempt")
            raise ToolInvocationAuthorizationError() if revoked else DomainUnavailableError()

    handler = _RecordingHandler()
    runtime = _registry(_spec(handler, max_attempts=3), recorder=Recorder()).bind(
        policy_name="manual_learning_loop", context=_run_context(), clock=lambda: 10.0
    )
    with pytest.raises(ToolCancelledError if revoked else ToolUnavailableError):
        await runtime.execute(_call())
    assert handler.inputs == []
    assert calls == (["reserve"] if phase == "reserve" else ["reserve", "start_attempt"])


@pytest.mark.parametrize("explicit", [False, True])
async def test_tool_cancellation_survives_failure_accounting_error(explicit):
    class Recorder(_FailureRecorder):
        async def fail(self, **kwargs):
            raise RuntimeError("ACCOUNTING-CANARY")

    cancellation = ToolCancelledError("cancelled") if explicit else asyncio.CancelledError()
    handler = _RecordingHandler([cancellation])
    runtime = _registry(_spec(handler), recorder=Recorder()).bind(
        policy_name="manual_learning_loop", context=_run_context(), clock=lambda: 10.0
    )
    with pytest.raises(type(cancellation)):
        await runtime.execute(_call())
    assert len(handler.inputs) == 1


async def test_read_only_success_accounting_unavailable_does_not_repeat_handler():
    from app.domain.errors import DomainUnavailableError

    class Recorder(_FailureRecorder):
        async def succeed(self, **kwargs):
            raise DomainUnavailableError(commit_outcome_unknown=True)

    handler = _RecordingHandler()
    runtime = _registry(_spec(handler, max_attempts=3), recorder=Recorder()).bind(
        policy_name="manual_learning_loop", context=_run_context(), clock=lambda: 10.0
    )
    with pytest.raises(ToolUnavailableError):
        await runtime.execute(_call())
    assert len(handler.inputs) == 1
