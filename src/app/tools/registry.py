from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from inspect import isawaitable
from math import isfinite
from time import monotonic
from typing import Literal, cast
from uuid import UUID, uuid4

from pydantic import JsonValue, ValidationError

from app.domain.action_execution import (
    ActionExecutionFailedError,
    ActionExecutionIdentity,
    ActionExecutionStore,
    ActionOutcomeUnknownError,
    ActionResultUnconfirmedError,
    ConfirmedActionResult,
    PreparedActionExecution,
)
from app.domain.actions import SUBMIT_APPLICATION_TOOL_NAME
from app.domain.errors import DomainUnavailableError
from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import (
    ToolInvocationAuthorizationError,
    ToolInvocationLimitError,
    ToolInvocationRecorderPort,
    ToolInvocationReservation,
)
from app.domain.tracing import (
    SpanStatus,
    TraceMetadataValue,
    bind_trace_scope,
    current_trace_scope,
    finish_trace_span,
    start_trace_span,
)
from app.llm.ports import ModelToolCall, ModelToolSchema
from app.mock_portal.contracts import MockSubmissionRequestV1, mock_submission_payload_digest
from app.tools.adapters.mock_portal import (
    MockPortalConflictError,
    MockPortalHTTPAdapter,
    MockPortalRejectedError,
    MockPortalTransportError,
)
from app.tools.contracts import (
    CancellationCheck,
    CredentialSource,
    GraphToolPolicy,
    InvocationIdFactory,
    MonotonicClock,
    ToolCallBudget,
    ToolExecutionContext,
    ToolInputModel,
    ToolOutputModel,
    ToolRunContext,
    ToolRuntime,
    ToolSpec,
    ToolTransientError,
)
from app.tools.invocations import (
    InMemoryToolInvocationRecorder,
    canonical_args_digest,
    output_summary,
)

_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def _approved_payload_digest(execution: PreparedActionExecution) -> str:
    return mock_submission_payload_digest(
        MockSubmissionRequestV1(
            workspace_id=execution.workspace_id,
            originating_actor_user_id=execution.originating_actor_user_id,
            run_id=execution.run_id,
            action_intent_id=execution.action_intent_id,
            payload=execution.args,
        )
    )


_RESERVED_INPUT_FIELDS = frozenset(
    {
        "workspace_id",
        "actor_user_id",
        "run_id",
        "document_id",
        "document_ids",
        "embedding_profile",
        "top_k",
        "context_budget",
        "role",
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
    }
)


class ToolRegistryError(Exception):
    """Base class for safe tool policy and execution failures."""


class ToolConfigurationError(ToolRegistryError):
    """Trusted tool or graph configuration is invalid."""


class ToolNotAllowedError(ToolRegistryError):
    """A model proposed a tool outside the current graph policy."""


class ToolInputValidationError(ToolRegistryError):
    """Model-provided business arguments failed the tool contract."""


class ToolCallLimitExceededError(ToolRegistryError):
    """The per-run logical call limit for a tool was exhausted."""


class ToolCancelledError(ToolRegistryError):
    """Execution was cancelled before a new tool attempt."""


class ToolDeadlineExceededError(ToolRegistryError):
    """The trusted monotonic deadline expired before a tool attempt."""


class ToolTimeoutError(ToolRegistryError):
    """All permitted timeout attempts were exhausted."""


class ToolUnavailableError(ToolRegistryError):
    """All permitted transient provider attempts were exhausted."""


class ToolExecutionError(ToolRegistryError):
    """A trusted handler failed without a valid output."""


class ToolOutputValidationError(ToolRegistryError):
    """A handler result failed its declared output contract."""


class ToolOutputTooLargeError(ToolRegistryError):
    """A valid handler result exceeded the model-visible output limit."""


class _HandlerRaisedTimeout(Exception):
    """Separate an explicit handler timeout from the registry timeout wrapper."""


@dataclass
class _ToolObservation:
    # Local observation counters only; never used for execution or retry decisions.
    attempts_started: int = 0
    output_bytes: int | None = None
    status: SpanStatus = "succeeded"


@contextmanager
def _observe_tool(
    tool_name: str, effect: ToolEffect, invocation_id: UUID
) -> Iterator[_ToolObservation]:
    scope = current_trace_scope()
    started_at = monotonic()
    context = (
        None
        if scope is None
        else start_trace_span(
            scope,
            span_kind="tool",
            metadata={
                "tool_name": tool_name,
                "tool_effect": effect.value,
                "tool_invocation_id": invocation_id,
            },
        )
    )
    observation = _ToolObservation()
    error_category = None
    # A dropped tool must not give retrieval its outer node as a substitute parent.
    child_scope = None if scope is None else replace(scope, parent=context)
    try:
        with bind_trace_scope(child_scope):
            yield observation
    except (asyncio.CancelledError, ToolCancelledError):
        observation.status, error_category = "cancelled", "cancelled"
        raise
    except Exception as error:
        observation.status = "failed"
        if isinstance(error, ToolDeadlineExceededError):
            error_category = "deadline_exceeded"
        elif isinstance(error, ToolTimeoutError):
            error_category = "provider_timeout"
        elif isinstance(error, ToolUnavailableError):
            error_category = "provider_unavailable"
        elif isinstance(error, ToolOutputValidationError | ToolOutputTooLargeError):
            error_category = "invalid_tool_output"
        elif isinstance(error, ActionExecutionFailedError):
            error_category = "action_execution_failed"
        elif isinstance(error, ActionOutcomeUnknownError):
            error_category = "action_outcome_unknown"
        elif isinstance(error, ActionResultUnconfirmedError):
            error_category = "action_result_unconfirmed"
        else:
            # Includes accounting failures; never inspect or export exception text.
            error_category = "tool_execution_failed"
        raise
    finally:
        metadata: dict[str, TraceMetadataValue] = {}
        if observation.attempts_started:
            metadata.update(
                attempt_number=observation.attempts_started,
                retry_count=max(observation.attempts_started - 1, 0),
            )
        if observation.output_bytes is not None:
            metadata["output_bytes"] = observation.output_bytes
        if scope is not None:
            finish_trace_span(
                scope,
                context,
                started_at=started_at,
                status=observation.status,
                error_category=error_category,
                metadata=metadata,
            )


ApprovedActionFaultPoint = Literal[
    "after_consume_commit_before_send_cas",
    "after_send_transition_commit_before_http",
    "after_recovery_get_found_before_success",
    "after_recovery_get_absent_before_resend",
    "after_recovery_resend_commit_before_http",
    "after_recovery_http_before_success",
]
type ApprovedActionFaultInjector = Callable[[ApprovedActionFaultPoint], Awaitable[None] | None]


def _configuration_error(message: str) -> ToolConfigurationError:
    return ToolConfigurationError(message)


def _is_positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _is_positive_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and isfinite(float(value))
        and float(value) > 0
    )


def _is_async_callable(value: object) -> bool:
    return callable(value) and (
        inspect.iscoroutinefunction(value) or inspect.iscoroutinefunction(type(value).__call__)
    )


def _validate_contract_model(
    model: object,
    *,
    expected_base: type[ToolInputModel] | type[ToolOutputModel],
) -> None:
    if not isinstance(model, type) or not issubclass(model, expected_base):
        raise _configuration_error("tool contract model uses the wrong base type")

    config = model.model_config
    if (
        config.get("extra") != "forbid"
        or config.get("frozen") is not True
        or config.get("strict") is not True
        or config.get("hide_input_in_errors") is not True
        or config.get("allow_inf_nan") is not False
    ):
        raise _configuration_error("tool contract model weakens the strict base policy")


def _validate_spec(spec: object) -> ToolSpec:
    if not isinstance(spec, ToolSpec):
        raise _configuration_error("registry entries must be ToolSpec values")
    if not isinstance(spec.name, str) or _TOOL_NAME_PATTERN.fullmatch(spec.name) is None:
        raise _configuration_error("tool names must use lower snake case")
    if not isinstance(spec.description, str) or not spec.description.strip():
        raise _configuration_error("tool descriptions must not be blank")

    _validate_contract_model(spec.input_model, expected_base=ToolInputModel)
    _validate_contract_model(spec.output_model, expected_base=ToolOutputModel)
    if _RESERVED_INPUT_FIELDS.intersection(spec.input_model.model_fields):
        raise _configuration_error("tool input schema declares a reserved execution field")
    if not isinstance(spec.effect, ToolEffect):
        raise _configuration_error("tool effect must use the authoritative enum")
    if not isinstance(spec.credential_source, CredentialSource):
        raise _configuration_error("tool credential source is invalid")
    if not _is_positive_finite_number(spec.timeout_seconds):
        raise _configuration_error("tool timeout must be a positive finite number")
    if not _is_positive_integer(spec.max_attempts):
        raise _configuration_error("tool max attempts must be a positive integer")
    if spec.max_attempts > 1 and spec.effect is not ToolEffect.READ_ONLY:
        raise _configuration_error("only read-only tools may retry")
    if not _is_positive_integer(spec.per_run_call_limit):
        raise _configuration_error("tool per-run call limit must be a positive integer")
    if not _is_positive_integer(spec.max_output_bytes):
        raise _configuration_error("tool output limit must be a positive integer")
    if not _is_async_callable(spec.handler):
        raise _configuration_error("tool handlers must be asynchronous callables")
    return spec


def _validate_policy(policy: object, specs: Mapping[str, ToolSpec]) -> GraphToolPolicy:
    if not isinstance(policy, GraphToolPolicy):
        raise _configuration_error("graph policies must be GraphToolPolicy values")
    if not isinstance(policy.name, str) or _TOOL_NAME_PATTERN.fullmatch(policy.name) is None:
        raise _configuration_error("graph policy names must use lower snake case")
    if not isinstance(policy.allowed_tool_names, frozenset) or not policy.allowed_tool_names:
        raise _configuration_error("graph tool allowlists must be non-empty frozensets")
    if not isinstance(policy.allowed_effects, frozenset) or not policy.allowed_effects:
        raise _configuration_error("graph effect allowlists must be non-empty frozensets")
    if any(
        not isinstance(name, str) or _TOOL_NAME_PATTERN.fullmatch(name) is None
        for name in policy.allowed_tool_names
    ):
        raise _configuration_error("graph tool allowlists contain an invalid name")
    if any(not isinstance(effect, ToolEffect) for effect in policy.allowed_effects):
        raise _configuration_error("graph effect allowlists contain an invalid effect")

    for name in policy.allowed_tool_names:
        spec = specs.get(name)
        if spec is None:
            raise _configuration_error("graph policy references an unregistered tool")
        if spec.effect not in policy.allowed_effects:
            raise _configuration_error("graph policy effect does not authorize its tool")
    return policy


def _model_schema(spec: ToolSpec) -> ModelToolSchema:
    try:
        schema = json.loads(json.dumps(spec.input_model.model_json_schema(mode="validation")))
        properties = schema.get("properties", {})
        if not isinstance(properties, dict) or _RESERVED_INPUT_FIELDS.intersection(properties):
            raise _configuration_error("tool input schema declares a reserved execution field")
        return ModelToolSchema(
            name=spec.name,
            description=spec.description,
            input_schema=cast(dict[str, JsonValue], schema),
        )
    except ToolConfigurationError:
        raise
    except Exception:
        raise _configuration_error("tool input schema generation failed") from None


class ToolRegistry:
    def __init__(
        self,
        *,
        specs: Iterable[ToolSpec],
        policies: Iterable[GraphToolPolicy],
        recorder: ToolInvocationRecorderPort | None = None,
        action_execution_store: ActionExecutionStore | None = None,
        approved_action_adapter: MockPortalHTTPAdapter | None = None,
        action_recovery_max_attempts: int = 3,
        approved_action_fault_injector: ApprovedActionFaultInjector | None = None,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        specs_by_name: dict[str, ToolSpec] = {}
        for candidate in specs:
            spec = _validate_spec(candidate)
            if spec.name in specs_by_name:
                raise _configuration_error("tool names must be unique")
            specs_by_name[spec.name] = spec
        if not specs_by_name:
            raise _configuration_error("tool registries must contain at least one tool")
        model_schemas = {name: _model_schema(spec) for name, spec in specs_by_name.items()}

        policies_by_name: dict[str, GraphToolPolicy] = {}
        for candidate in policies:
            policy = _validate_policy(candidate, specs_by_name)
            if policy.name in policies_by_name:
                raise _configuration_error("graph policy names must be unique")
            policies_by_name[policy.name] = policy
        if not policies_by_name:
            raise _configuration_error("tool registries must contain at least one graph policy")

        self._specs = specs_by_name
        self._model_schemas = model_schemas
        self._policies = policies_by_name
        self._recorder = recorder
        self._action_execution_store = action_execution_store
        self._approved_action_adapter = approved_action_adapter
        if not _is_positive_integer(action_recovery_max_attempts):
            raise _configuration_error("action recovery max attempts must be positive")
        self._action_recovery_max_attempts = action_recovery_max_attempts
        self._approved_action_fault_injector = approved_action_fault_injector
        self._wall_clock = wall_clock

    def bind(
        self,
        *,
        policy_name: str,
        context: ToolRunContext,
        invocation_id_factory: InvocationIdFactory = uuid4,
        clock: MonotonicClock = monotonic,
    ) -> ToolRuntime:
        policy = self._policies.get(policy_name)
        if policy is None:
            raise _configuration_error("requested graph policy is not registered")
        _validate_run_context(context)
        if not callable(invocation_id_factory) or not callable(clock):
            raise _configuration_error("runtime factories must be callable")

        return _BoundToolRuntime(
            specs=self._specs,
            model_schemas=self._model_schemas,
            policy=policy,
            context=_copy_run_context(context),
            invocation_id_factory=invocation_id_factory,
            clock=clock,
            recorder=self._recorder or InMemoryToolInvocationRecorder(),
        )

    async def _inject_approved_action(self, point: ApprovedActionFaultPoint) -> None:
        if self._approved_action_fault_injector is None:
            return
        result = self._approved_action_fault_injector(point)
        if isawaitable(result):
            await result

    async def execute_approved_action(
        self,
        identity: ActionExecutionIdentity,
        *,
        deadline: float,
        cancellation: CancellationCheck,
        clock: MonotonicClock = monotonic,
    ) -> str:
        """Execute one persisted irreversible action without a model tool call."""
        if self._action_execution_store is None or self._approved_action_adapter is None:
            raise ToolConfigurationError("approved action execution store is not configured")
        spec = self._specs.get(SUBMIT_APPLICATION_TOOL_NAME)
        if (
            spec is None
            or spec.effect is not ToolEffect.IRREVERSIBLE
            or spec.max_attempts != 1
            or spec.per_run_call_limit != 1
        ):
            raise ToolConfigurationError("approved action tool policy is invalid")
        if not isinstance(identity, ActionExecutionIdentity):
            raise ToolInputValidationError("approved action identity is invalid")
        if not isinstance(cancellation, CancellationCheck):
            raise ToolConfigurationError("approved action cancellation guard is invalid")
        start = float(clock())
        if deadline <= start:
            raise ToolDeadlineExceededError("approved action deadline has expired")

        prepared = await self._action_execution_store.prepare_execution(
            identity, now=self._wall_clock()
        )
        with _observe_tool(prepared.tool_name, spec.effect, prepared.invocation_id) as observation:
            if prepared.status == "succeeded":
                observation.status = "skipped"
                external_ref = (
                    None if prepared.result is None else prepared.result.get("external_ref")
                )
                if not isinstance(external_ref, str) or not external_ref:
                    raise ToolExecutionError("persisted successful action result is invalid")
                normalized = _normalize_output(
                    spec,
                    spec.output_model.model_validate({"external_ref": external_ref}, strict=True),
                )
                observation.output_bytes = len(normalized.encode("utf-8"))
                return normalized
            if prepared.status == "failed":
                raise ActionExecutionFailedError
            if prepared.status == "outcome_unknown":
                raise ActionOutcomeUnknownError
            if prepared.status == "executing":
                normalized = await self._reconcile_approved_action(
                    identity=identity,
                    prepared=prepared,
                    spec=spec,
                    deadline=deadline,
                    cancellation=cancellation,
                    start=start,
                    clock=clock,
                )
                observation.output_bytes = len(normalized.encode("utf-8"))
                return normalized
            await self._inject_approved_action("after_consume_commit_before_send_cas")
            authorization = await self._action_execution_store.begin_send(
                identity, now=self._wall_clock()
            )
            if not authorization.allowed:
                raise ToolCancelledError(
                    "approved action was cancelled or authorization was revoked before send"
                )
            await self._inject_approved_action("after_send_transition_commit_before_http")

            if (
                authorization.execution.workspace_id != prepared.workspace_id
                or authorization.execution.originating_actor_user_id
                != prepared.originating_actor_user_id
                or authorization.execution.run_id != prepared.run_id
                or authorization.execution.action_intent_id != prepared.action_intent_id
                or authorization.execution.approval_request_id != prepared.approval_request_id
                or authorization.execution.invocation_id != prepared.invocation_id
                or authorization.execution.tool_name != prepared.tool_name
                or authorization.execution.args != prepared.args
                or authorization.execution.target != prepared.target
                or authorization.execution.idempotency_key != prepared.idempotency_key
            ):
                raise ToolExecutionError("approved action execution facts changed before send")
            try:
                tool_input = spec.input_model.model_validate(
                    prepared.args.model_dump(mode="python", round_trip=True), strict=True
                )
            except ValidationError:
                raise ToolExecutionError("persisted approved action args are invalid") from None
            context = ToolExecutionContext(
                workspace_id=prepared.workspace_id,
                actor_user_id=prepared.originating_actor_user_id,
                run_id=prepared.run_id,
                invocation_id=prepared.invocation_id,
                action_intent_id=prepared.action_intent_id,
                approval_request_id=prepared.approval_request_id,
                trusted_target=prepared.target.model_dump(mode="json", round_trip=True),
                deadline=deadline,
                budget=ToolCallBudget(call_number=1, call_limit=1, remaining_calls=0),
                cancellation=cancellation,
            )
            remaining = deadline - float(clock())
            if remaining <= 0:
                raise ToolDeadlineExceededError("approved action deadline expired before HTTP send")
            try:
                async with asyncio.timeout(min(float(spec.timeout_seconds), remaining)):
                    observation.attempts_started = 1
                    output = await spec.handler(tool_input, context)
            except asyncio.CancelledError:
                raise
            except MockPortalRejectedError:
                latency_ms = max(0, round((float(clock()) - start) * 1000))
                await self._action_execution_store.confirm_failure(
                    identity,
                    error_category="external_rejected_no_effect",
                    evidence={"classification": "explicit_no_effect_rejection"},
                    latency_ms=latency_ms,
                    now=self._wall_clock(),
                )
                raise ActionExecutionFailedError from None
            except (MockPortalTransportError, MockPortalConflictError):
                raise ActionResultUnconfirmedError from None
            except Exception as error:
                if isinstance(error, ToolRegistryError):
                    raise
                raise ActionResultUnconfirmedError from None
            normalized = _normalize_output(spec, output)
            try:
                output_value = spec.output_model.model_validate(output, strict=True)
                external_ref = output_value.model_dump(mode="python")["external_ref"]
            except (KeyError, TypeError, ValidationError):
                raise ToolOutputValidationError("approved action output is invalid") from None
            if not isinstance(external_ref, str):
                raise ToolOutputValidationError("approved action external ref is invalid")
            latency_ms = max(0, round((float(clock()) - start) * 1000))
            await self._confirm_action_success(
                identity,
                result=ConfirmedActionResult(
                    external_ref=external_ref,
                    payload_digest=_approved_payload_digest(authorization.execution),
                    source="initial_response",
                ),
                latency_ms=latency_ms,
                now=self._wall_clock(),
            )
            observation.output_bytes = len(normalized.encode("utf-8"))
            return normalized

    async def _confirm_action_success(
        self,
        identity: ActionExecutionIdentity,
        *,
        result: ConfirmedActionResult,
        latency_ms: int,
        now: datetime,
    ) -> None:
        try:
            await self._action_execution_store.confirm_success(
                identity, result=result, latency_ms=latency_ms, now=now
            )
        except Exception:
            # External evidence exists even if its database commit was not acknowledged.
            # Recovery must inspect durable state and reconcile, never blindly POST.
            raise ActionResultUnconfirmedError from None

    async def _reconcile_approved_action(
        self,
        *,
        identity: ActionExecutionIdentity,
        prepared: PreparedActionExecution,
        spec: ToolSpec,
        deadline: float,
        cancellation: CancellationCheck,
        start: float,
        clock: MonotonicClock,
    ) -> str:
        if self._action_execution_store is None or self._approved_action_adapter is None:
            raise ToolConfigurationError("approved action recovery is not configured")
        authorization = await self._action_execution_store.begin_reconciliation(
            identity,
            max_attempts=self._action_recovery_max_attempts,
            now=self._wall_clock(),
        )
        if not authorization.network_allowed:
            raise ActionOutcomeUnknownError
        execution = authorization.execution
        remaining = deadline - float(clock())
        if remaining <= 0:
            await self._action_execution_store.confirm_outcome_unknown(
                identity,
                evidence={"classification": "reconciliation_deadline_unavailable"},
                latency_ms=max(0, round((float(clock()) - start) * 1000)),
                now=self._wall_clock(),
            )
            raise ActionOutcomeUnknownError
        try:
            async with asyncio.timeout(min(float(spec.timeout_seconds), remaining)):
                evidence = await self._approved_action_adapter.get_by_idempotency_key(
                    workspace_id=execution.workspace_id,
                    actor_user_id=execution.originating_actor_user_id,
                    run_id=execution.run_id,
                    action_intent_id=execution.action_intent_id,
                    idempotency_key=execution.idempotency_key,
                    payload=execution.args,
                )
        except asyncio.CancelledError:
            raise
        except (MockPortalTransportError, TimeoutError):
            await self._action_execution_store.confirm_outcome_unknown(
                identity,
                evidence={
                    "classification": "reconciliation_query_unavailable_or_contradictory",
                    "recovery_attempt": authorization.recovery_attempt,
                },
                latency_ms=max(0, round((float(clock()) - start) * 1000)),
                now=self._wall_clock(),
            )
            raise ActionOutcomeUnknownError from None
        if evidence is not None:
            await self._inject_approved_action("after_recovery_get_found_before_success")
            await self._confirm_action_success(
                identity,
                result=ConfirmedActionResult(
                    external_ref=evidence.external_ref,
                    payload_digest=evidence.payload_digest,
                    source="reconciliation_query",
                ),
                latency_ms=max(0, round((float(clock()) - start) * 1000)),
                now=self._wall_clock(),
            )
            return _normalize_output(
                spec,
                spec.output_model.model_validate(
                    {"external_ref": evidence.external_ref}, strict=True
                ),
            )

        await self._inject_approved_action("after_recovery_get_absent_before_resend")
        resend = await self._action_execution_store.authorize_resend(
            identity,
            expected_recovery_attempt=authorization.recovery_attempt,
            now=self._wall_clock(),
        )
        if not resend.allowed:
            raise ActionExecutionFailedError
        await self._inject_approved_action("after_recovery_resend_commit_before_http")
        execution = resend.execution
        try:
            remaining = deadline - float(clock())
            if remaining <= 0:
                raise MockPortalTransportError
            async with asyncio.timeout(min(float(spec.timeout_seconds), remaining)):
                external_ref = await self._approved_action_adapter.submit(
                    workspace_id=execution.workspace_id,
                    actor_user_id=execution.originating_actor_user_id,
                    run_id=execution.run_id,
                    action_intent_id=execution.action_intent_id,
                    idempotency_key=execution.idempotency_key,
                    payload=execution.args,
                )
        except asyncio.CancelledError:
            raise
        except MockPortalRejectedError:
            await self._action_execution_store.confirm_failure(
                identity,
                error_category="external_rejected_no_effect",
                evidence={
                    "classification": "explicit_no_effect_rejection",
                    "recovery_attempt": authorization.recovery_attempt,
                },
                latency_ms=max(0, round((float(clock()) - start) * 1000)),
                now=self._wall_clock(),
            )
            raise ActionExecutionFailedError from None
        except (MockPortalTransportError, MockPortalConflictError, TimeoutError):
            raise ActionResultUnconfirmedError from None
        await self._inject_approved_action("after_recovery_http_before_success")
        await self._confirm_action_success(
            identity,
            result=ConfirmedActionResult(
                external_ref=external_ref,
                payload_digest=_approved_payload_digest(execution),
                source="recovery_resend",
            ),
            latency_ms=max(0, round((float(clock()) - start) * 1000)),
            now=self._wall_clock(),
        )
        return _normalize_output(
            spec, spec.output_model.model_validate({"external_ref": external_ref}, strict=True)
        )


def _validate_optional_uuid(value: object) -> bool:
    return value is None or isinstance(value, UUID)


def _validate_run_context(context: object) -> ToolRunContext:
    if not isinstance(context, ToolRunContext):
        raise _configuration_error("tool run context uses the wrong contract")
    if not all(
        isinstance(value, UUID)
        for value in (context.workspace_id, context.actor_user_id, context.run_id)
    ):
        raise _configuration_error("tool run identity fields must be UUID values")
    if not _validate_optional_uuid(context.action_intent_id) or not _validate_optional_uuid(
        context.approval_request_id
    ):
        raise _configuration_error("optional tool action identity fields must be UUID values")
    if context.trusted_target is not None and not isinstance(context.trusted_target, Mapping):
        raise _configuration_error("trusted tool target must be a mapping")
    if (
        isinstance(context.deadline, bool)
        or not isinstance(context.deadline, int | float)
        or not isfinite(float(context.deadline))
    ):
        raise _configuration_error("tool deadline must be a finite monotonic timestamp")
    if not isinstance(context.cancellation, CancellationCheck):
        raise _configuration_error("tool cancellation check uses the wrong contract")
    return context


def _copy_run_context(context: ToolRunContext) -> ToolRunContext:
    trusted_target: dict[str, JsonValue] | None = None
    if context.trusted_target is not None:
        try:
            candidate = deepcopy(dict(context.trusted_target))
            if any(not isinstance(key, str) for key in candidate):
                raise TypeError
            json.dumps(candidate, allow_nan=False)
            trusted_target = cast(dict[str, JsonValue], candidate)
        except Exception:
            raise _configuration_error("trusted tool target must contain JSON values") from None

    return ToolRunContext(
        workspace_id=context.workspace_id,
        actor_user_id=context.actor_user_id,
        run_id=context.run_id,
        action_intent_id=context.action_intent_id,
        approval_request_id=context.approval_request_id,
        trusted_target=trusted_target,
        deadline=float(context.deadline),
        cancellation=context.cancellation,
    )


class _BoundToolRuntime:
    def __init__(
        self,
        *,
        specs: Mapping[str, ToolSpec],
        model_schemas: Mapping[str, ModelToolSchema],
        policy: GraphToolPolicy,
        context: ToolRunContext,
        invocation_id_factory: InvocationIdFactory,
        clock: MonotonicClock,
        recorder: ToolInvocationRecorderPort,
    ) -> None:
        self._specs = dict(specs)
        self._policy = policy
        self._context = context
        self._invocation_id_factory = invocation_id_factory
        self._clock = clock
        self._recorder = recorder
        self._model_tools = tuple(
            model_schemas[name].model_copy(deep=True) for name in sorted(policy.allowed_tool_names)
        )

    def model_tools(self) -> tuple[ModelToolSchema, ...]:
        return tuple(tool.model_copy(deep=True) for tool in self._model_tools)

    def _validated_call(self, call: ModelToolCall) -> tuple[ToolSpec, ToolInputModel]:
        if not isinstance(call, ModelToolCall):
            raise ToolInputValidationError("tool call uses the wrong contract")
        if call.name not in self._policy.allowed_tool_names:
            raise ToolNotAllowedError("tool is not allowed for the current graph")

        spec = self._specs[call.name]
        if spec.effect not in self._policy.allowed_effects:
            raise ToolNotAllowedError("tool effect is not allowed for the current graph")
        if _RESERVED_INPUT_FIELDS.intersection(call.arguments):
            raise ToolInputValidationError("tool call contains a reserved execution field")

        try:
            tool_input = spec.input_model.model_validate(deepcopy(call.arguments), strict=True)
        except ValidationError:
            raise ToolInputValidationError("tool call arguments failed validation") from None

        return spec, tool_input

    def validate_call(self, call: ModelToolCall) -> None:
        self._validated_call(call)

    async def execute(self, call: ModelToolCall) -> str:
        spec, tool_input = self._validated_call(call)

        self._raise_if_cancelled()
        self._remaining_time()
        reservation = await self._reserve_call(spec, call.arguments)
        context = ToolExecutionContext(
            workspace_id=self._context.workspace_id,
            actor_user_id=self._context.actor_user_id,
            run_id=self._context.run_id,
            invocation_id=reservation.invocation_id,
            action_intent_id=self._context.action_intent_id,
            approval_request_id=self._context.approval_request_id,
            trusted_target=self._context.trusted_target,
            deadline=self._context.deadline,
            budget=ToolCallBudget(
                call_number=reservation.call_number,
                call_limit=spec.per_run_call_limit,
                remaining_calls=spec.per_run_call_limit - reservation.call_number,
            ),
            cancellation=self._context.cancellation,
        )

        with _observe_tool(spec.name, spec.effect, reservation.invocation_id) as observation:
            normalized = await self._invoke_and_normalize(
                spec, tool_input, context, reservation, observation
            )
            observation.output_bytes = len(normalized.encode("utf-8"))
            return normalized

    async def _reserve_call(
        self,
        spec: ToolSpec,
        arguments: dict[str, JsonValue],
    ) -> ToolInvocationReservation:
        try:
            invocation_id = self._invocation_id_factory()
        except Exception:
            raise _configuration_error("invocation id factory failed") from None
        if not isinstance(invocation_id, UUID):
            raise _configuration_error("invocation id factory must return UUID values")
        try:
            return await self._recorder.reserve(
                invocation_id=invocation_id,
                workspace_id=self._context.workspace_id,
                actor_user_id=self._context.actor_user_id,
                run_id=self._context.run_id,
                tool_name=spec.name,
                effect=spec.effect,
                args_digest=canonical_args_digest(arguments),
                call_limit=spec.per_run_call_limit,
            )
        except ToolInvocationLimitError:
            raise ToolCallLimitExceededError("tool per-run call limit was exceeded") from None
        except ToolInvocationAuthorizationError:
            raise ToolCancelledError("tool execution authorization was revoked") from None
        except DomainUnavailableError:
            raise ToolUnavailableError("tool invocation storage is unavailable") from None
        except Exception:
            raise ToolExecutionError("tool invocation accounting failed") from None

    def _clock_value(self) -> float:
        try:
            value = self._clock()
        except Exception:
            raise _configuration_error("monotonic clock failed") from None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(float(value))
        ):
            raise _configuration_error("monotonic clock must return a finite number")
        return float(value)

    def _remaining_time(self) -> float:
        remaining = self._context.deadline - self._clock_value()
        if remaining <= 0:
            raise ToolDeadlineExceededError("tool execution deadline has expired")
        return remaining

    def _raise_if_cancelled(self) -> None:
        try:
            cancelled = self._context.cancellation.is_cancelled()
        except Exception:
            raise _configuration_error("tool cancellation check failed") from None
        if not isinstance(cancelled, bool):
            raise _configuration_error("tool cancellation check must return a boolean")
        if cancelled:
            raise ToolCancelledError("tool execution was cancelled before send")

    async def _invoke_and_normalize(
        self,
        spec: ToolSpec,
        tool_input: ToolInputModel,
        context: ToolExecutionContext,
        reservation: ToolInvocationReservation,
        observation: _ToolObservation,
    ) -> str:
        started_at = self._clock_value()
        for attempt in range(1, spec.max_attempts + 1):
            try:
                self._raise_if_cancelled()
                remaining_time = self._remaining_time()
                timeout_seconds = min(float(spec.timeout_seconds), remaining_time)
                deadline_limits_attempt = remaining_time <= float(spec.timeout_seconds)
                await self._recorder.start_attempt(
                    reservation=reservation,
                    workspace_id=context.workspace_id,
                    actor_user_id=context.actor_user_id,
                    run_id=context.run_id,
                    attempt=attempt,
                )
                observation.attempts_started += 1
                async with asyncio.timeout(timeout_seconds):
                    try:
                        output = await spec.handler(tool_input, context)
                    except TimeoutError:
                        raise _HandlerRaisedTimeout from None
            except (asyncio.CancelledError, ToolCancelledError) as cancellation:
                try:
                    await self._record_failure_durably(
                        reservation,
                        started_at=started_at,
                        error_category="cancelled",
                    )
                except Exception:
                    raise cancellation from None
                raise
            except ToolDeadlineExceededError:
                await self._record_failure_durably(
                    reservation,
                    started_at=started_at,
                    error_category="deadline_exceeded",
                )
                raise
            except ToolInvocationAuthorizationError:
                try:
                    await self._record_failure_durably(
                        reservation,
                        started_at=started_at,
                        error_category="cancelled",
                    )
                except Exception:
                    raise ToolCancelledError("tool execution authorization was revoked") from None
                raise ToolCancelledError("tool execution authorization was revoked") from None
            except _HandlerRaisedTimeout:
                if attempt < spec.max_attempts:
                    continue
                await self._record_failure_durably(
                    reservation,
                    started_at=started_at,
                    error_category="provider_timeout",
                )
                raise ToolTimeoutError("tool timeout attempts were exhausted") from None
            except TimeoutError:
                if deadline_limits_attempt:
                    await self._record_failure_durably(
                        reservation,
                        started_at=started_at,
                        error_category="deadline_exceeded",
                    )
                    raise ToolDeadlineExceededError(
                        "tool execution deadline expired during an attempt"
                    ) from None
                if attempt < spec.max_attempts:
                    continue
                await self._record_failure_durably(
                    reservation,
                    started_at=started_at,
                    error_category="provider_timeout",
                )
                raise ToolTimeoutError("tool timeout attempts were exhausted") from None
            except DomainUnavailableError:
                # Storage failure is not a provider failure; do not retry inside the tool loop.
                raise ToolUnavailableError("tool invocation storage is unavailable") from None
            except ToolTransientError:
                if attempt < spec.max_attempts:
                    continue
                await self._record_failure_durably(
                    reservation,
                    started_at=started_at,
                    error_category="provider_unavailable",
                )
                raise ToolUnavailableError("tool transient attempts were exhausted") from None
            except Exception:
                await self._record_failure_durably(
                    reservation,
                    started_at=started_at,
                    error_category="tool_execution_failed",
                )
                raise ToolExecutionError("tool handler failed") from None

            try:
                normalized = _normalize_output(spec, output)
            except ToolRegistryError:
                await self._record_failure_durably(
                    reservation,
                    started_at=started_at,
                    error_category="invalid_tool_output",
                )
                raise
            try:
                await self._recorder.succeed(
                    reservation=reservation,
                    workspace_id=context.workspace_id,
                    run_id=context.run_id,
                    latency_ms=self._elapsed_ms(started_at),
                    result_summary=output_summary(normalized),
                )
            except DomainUnavailableError:
                raise ToolUnavailableError("tool invocation storage is unavailable") from None
            return normalized

        raise AssertionError("tool attempt loop exhausted without returning or raising")

    def _elapsed_ms(self, started_at: float) -> int:
        elapsed = self._clock_value() - started_at
        if elapsed < 0:
            raise _configuration_error("monotonic clock moved backwards")
        return round(elapsed * 1000)

    async def _record_failure_durably(
        self,
        reservation: ToolInvocationReservation,
        *,
        started_at: float,
        error_category: str,
    ) -> None:
        finalize_task = asyncio.create_task(
            self._record_failure(
                reservation,
                started_at=started_at,
                error_category=error_category,
            )
        )
        try:
            await asyncio.shield(finalize_task)
        except asyncio.CancelledError:
            await finalize_task
            raise

    async def _record_failure(
        self,
        reservation: ToolInvocationReservation,
        *,
        started_at: float,
        error_category: str,
    ) -> None:
        try:
            await self._recorder.fail(
                reservation=reservation,
                workspace_id=self._context.workspace_id,
                run_id=self._context.run_id,
                latency_ms=self._elapsed_ms(started_at),
                error_category=error_category,
            )
        except DomainUnavailableError:
            raise ToolUnavailableError("tool invocation storage is unavailable") from None
        except Exception:
            raise ToolExecutionError("tool invocation accounting failed") from None


def _normalize_output(spec: ToolSpec, output: object) -> str:
    try:
        validated = spec.output_model.model_validate(output, strict=True)
        payload = validated.model_dump(mode="json", round_trip=True)
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, ValidationError):
        raise ToolOutputValidationError("tool output failed validation") from None

    if len(serialized.encode("utf-8")) > spec.max_output_bytes:
        raise ToolOutputTooLargeError("tool output exceeded its byte limit")
    return serialized
