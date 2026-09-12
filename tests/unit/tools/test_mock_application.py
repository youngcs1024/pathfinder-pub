from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

import httpx
import pytest

from app.domain.action_execution import (
    ActionExecutionIdentity,
    ActionOutcomeUnknownError,
    ActionResultUnconfirmedError,
    ConfirmedActionResult,
    PreparedActionExecution,
    ReconciliationAuthorization,
    ResendAuthorization,
    SendAuthorization,
)
from app.domain.actions import SubmitApplicationArgsV1, TrustedActionTargetV1
from app.mock_portal.contracts import MockSubmissionRequestV1, mock_submission_payload_digest
from app.tools.adapters.mock_portal import MockPortalHTTPAdapter
from app.tools.contracts import ToolCallBudget, ToolExecutionContext
from app.tools.mock_application import (
    SubmitMockApplicationHandler,
    SubmitMockApplicationInput,
    create_approved_action_registry,
)


class _Cancellation:
    def is_cancelled(self) -> bool:
        return False


class _ExecutionStore:
    def __init__(self, prepared: PreparedActionExecution) -> None:
        self.prepared = prepared
        self.confirmed: ConfirmedActionResult | None = None
        self.prepare_count = 0
        self.begin_count = 0

    async def prepare_execution(self, identity, *, now):
        self.prepare_count += 1
        return self.prepared

    async def begin_send(self, identity, *, now):
        self.begin_count += 1
        return SendAuthorization(execution=self.prepared, allowed=True)

    async def confirm_success(self, identity, *, result, latency_ms, now):
        self.confirmed = result


class _RecoveryStore(_ExecutionStore):
    def __init__(self, prepared: PreparedActionExecution, *, network_allowed: bool = True) -> None:
        super().__init__(replace(prepared, status="executing"))
        self.network_allowed = network_allowed
        self.reconciliation_count = 0
        self.resend_count = 0
        self.unknown_evidence = None

    async def begin_reconciliation(self, identity, *, max_attempts, now):
        self.reconciliation_count += 1
        execution = replace(self.prepared, recovery_attempts=self.reconciliation_count)
        return ReconciliationAuthorization(
            execution=execution,
            recovery_attempt=self.reconciliation_count,
            network_allowed=self.network_allowed,
        )

    async def authorize_resend(self, identity, *, expected_recovery_attempt, now):
        self.resend_count += 1
        return ResendAuthorization(
            execution=replace(self.prepared, recovery_attempts=expected_recovery_attempt),
            recovery_attempt=expected_recovery_attempt,
            allowed=True,
        )

    async def confirm_outcome_unknown(self, identity, *, evidence, latency_ms, now):
        self.unknown_evidence = evidence


def _prepared() -> PreparedActionExecution:
    action_id = uuid4()
    return PreparedActionExecution(
        workspace_id=uuid4(),
        originating_actor_user_id=uuid4(),
        run_id=uuid4(),
        action_intent_id=action_id,
        approval_request_id=uuid4(),
        invocation_id=uuid4(),
        tool_name="submit_mock_application",
        args=SubmitApplicationArgsV1(
            job_ref="persisted-job",
            resume_document_id=uuid4(),
            answers={"availability": "two weeks"},
            cover_letter="Persisted exact draft",
        ),
        target=TrustedActionTargetV1(),
        idempotency_key=str(action_id),
    )


async def test_registry_trusted_path_uses_only_persisted_facts_and_one_http_attempt() -> None:
    prepared = _prepared()
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        digest = mock_submission_payload_digest(
            MockSubmissionRequestV1(
                workspace_id=prepared.workspace_id,
                originating_actor_user_id=prepared.originating_actor_user_id,
                run_id=prepared.run_id,
                action_intent_id=prepared.action_intent_id,
                payload=prepared.args,
            )
        )
        return httpx.Response(
            201,
            json={
                "id": str(uuid4()),
                "workspace_id": str(prepared.workspace_id),
                "originating_actor_user_id": str(prepared.originating_actor_user_id),
                "run_id": str(prepared.run_id),
                "action_intent_id": str(prepared.action_intent_id),
                "idempotency_key": prepared.idempotency_key,
                "payload_digest": digest,
                "external_ref": "mock-submission:stable",
                "created_at": datetime.now(UTC).isoformat(),
                "created": True,
            },
        )

    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        store = _ExecutionStore(prepared)
        registry = create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(client), action_execution_store=store
        )
        identity = ActionExecutionIdentity(
            prepared.workspace_id,
            prepared.run_id,
            prepared.action_intent_id,
            prepared.approval_request_id,
        )
        result = await registry.execute_approved_action(
            identity,
            deadline=monotonic() + 10,
            cancellation=_Cancellation(),
        )

    assert result == '{"external_ref":"mock-submission:stable"}'
    assert len(requests) == 1
    assert requests[0].headers["Idempotency-Key"] == str(prepared.action_intent_id)
    assert b"Persisted exact draft" in requests[0].content
    assert store.prepare_count == store.begin_count == 1
    assert store.confirmed is not None
    assert store.confirmed.external_ref == "mock-submission:stable"
    assert store.confirmed.source == "initial_response"


async def test_direct_handler_without_trusted_action_context_fails_closed() -> None:
    async with httpx.AsyncClient(base_url="http://mock.test") as client:
        handler = SubmitMockApplicationHandler(MockPortalHTTPAdapter(client))
        prepared = _prepared()
        tool_input = SubmitMockApplicationInput.model_validate(
            prepared.args.model_dump(mode="python"), strict=True
        )
        context = ToolExecutionContext(
            workspace_id=prepared.workspace_id,
            actor_user_id=prepared.originating_actor_user_id,
            run_id=prepared.run_id,
            invocation_id=prepared.invocation_id,
            action_intent_id=None,
            approval_request_id=None,
            trusted_target=None,
            deadline=monotonic() + 10,
            budget=ToolCallBudget(call_number=1, call_limit=1, remaining_calls=0),
            cancellation=_Cancellation(),
        )
        with pytest.raises(PermissionError, match="trusted approved action context"):
            await handler(tool_input, context)


@pytest.mark.parametrize(
    ("fault_point", "expected_prepare", "expected_begin", "expected_http"),
    [
        ("after_consume_commit_before_send_cas", 1, 0, 0),
        ("after_send_transition_commit_before_http", 1, 1, 0),
    ],
)
async def test_registry_fault_barriers_stop_before_the_next_external_boundary(
    fault_point: str,
    expected_prepare: int,
    expected_begin: int,
    expected_http: int,
) -> None:
    prepared = _prepared()
    http_count = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal http_count
        http_count += 1
        raise AssertionError("HTTP must not be reached")

    def fail(point: str) -> None:
        if point == fault_point:
            raise RuntimeError("injected crash")

    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        store = _ExecutionStore(prepared)
        registry = create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(client),
            action_execution_store=store,
            fault_injector=fail,
        )
        identity = ActionExecutionIdentity(
            prepared.workspace_id,
            prepared.run_id,
            prepared.action_intent_id,
            prepared.approval_request_id,
        )
        with pytest.raises(RuntimeError, match="injected crash"):
            await registry.execute_approved_action(
                identity,
                deadline=monotonic() + 10,
                cancellation=_Cancellation(),
            )
    assert store.prepare_count == expected_prepare
    assert store.begin_count == expected_begin
    assert http_count == expected_http


def _mock_response(prepared: PreparedActionExecution, *, created: bool) -> dict[str, object]:
    digest = mock_submission_payload_digest(
        MockSubmissionRequestV1(
            workspace_id=prepared.workspace_id,
            originating_actor_user_id=prepared.originating_actor_user_id,
            run_id=prepared.run_id,
            action_intent_id=prepared.action_intent_id,
            payload=prepared.args,
        )
    )
    return {
        "id": str(uuid4()),
        "workspace_id": str(prepared.workspace_id),
        "originating_actor_user_id": str(prepared.originating_actor_user_id),
        "run_id": str(prepared.run_id),
        "action_intent_id": str(prepared.action_intent_id),
        "idempotency_key": prepared.idempotency_key,
        "payload_digest": digest,
        "external_ref": "mock-submission:recovered",
        "created_at": datetime.now(UTC).isoformat(),
        "created": created,
    }


def _identity(prepared: PreparedActionExecution) -> ActionExecutionIdentity:
    return ActionExecutionIdentity(
        prepared.workspace_id,
        prepared.run_id,
        prepared.action_intent_id,
        prepared.approval_request_id,
    )


async def test_recovery_get_found_confirms_without_post() -> None:
    prepared = _prepared()
    methods: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=_mock_response(prepared, created=False))

    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        store = _RecoveryStore(prepared)
        registry = create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(client), action_execution_store=store
        )
        result = await registry.execute_approved_action(
            _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
        )

    assert result == '{"external_ref":"mock-submission:recovered"}'
    assert methods == ["GET"]
    assert store.resend_count == 0
    assert store.confirmed is not None
    assert store.confirmed.external_ref == "mock-submission:recovered"
    assert store.confirmed.source == "reconciliation_query"


async def test_recovery_explicit_absence_authorizes_same_key_resend() -> None:
    prepared = _prepared()
    methods: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(404)
        assert request.headers["Idempotency-Key"] == prepared.idempotency_key
        return httpx.Response(201, json=_mock_response(prepared, created=True))

    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        store = _RecoveryStore(prepared)
        registry = create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(client), action_execution_store=store
        )
        await registry.execute_approved_action(
            _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
        )

    assert methods == ["GET", "POST"]
    assert store.resend_count == 1
    assert store.confirmed is not None
    assert store.confirmed.external_ref == "mock-submission:recovered"
    assert store.confirmed.source == "recovery_resend"


@pytest.mark.parametrize(
    "response_kind",
    [
        "500",
        "connect_error",
        "timeout",
        "malformed",
        "workspace_id",
        "originating_actor_user_id",
        "run_id",
        "action_intent_id",
        "idempotency_key",
        "payload_digest",
    ],
)
async def test_recovery_unavailable_or_contradictory_becomes_unknown(
    response_kind: str,
) -> None:
    prepared = _prepared()

    def respond(request: httpx.Request) -> httpx.Response:
        if response_kind == "500":
            return httpx.Response(500)
        if response_kind == "connect_error":
            raise httpx.ConnectError("unavailable", request=request)
        if response_kind == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if response_kind == "malformed":
            return httpx.Response(200, content=b"not-json")
        body = _mock_response(prepared, created=False)
        body[response_kind] = (
            f"sha256:{'f' * 64}" if response_kind == "payload_digest" else str(uuid4())
        )
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        store = _RecoveryStore(prepared)
        registry = create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(client), action_execution_store=store
        )
        with pytest.raises(ActionOutcomeUnknownError):
            await registry.execute_approved_action(
                _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
            )

    assert store.resend_count == 0
    assert store.unknown_evidence == {
        "classification": "reconciliation_query_unavailable_or_contradictory",
        "recovery_attempt": 1,
    }


async def test_recovery_exhaustion_performs_no_network_io() -> None:
    prepared = _prepared()
    http_count = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal http_count
        http_count += 1
        raise AssertionError("recovery exhaustion must not access the target")

    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        store = _RecoveryStore(prepared, network_allowed=False)
        registry = create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(client), action_execution_store=store
        )
        with pytest.raises(ActionOutcomeUnknownError):
            await registry.execute_approved_action(
                _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
            )
    assert http_count == 0


async def test_response_loss_recovers_by_get_without_second_post() -> None:
    from tests.tracing import CollectingTraceSink, collecting_node

    sink = CollectingTraceSink()
    prepared = _prepared()
    post_count = 0
    get_count = 0
    target_exists = False

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal get_count, post_count, target_exists
        if request.method == "POST":
            post_count += 1
            target_exists = True
            raise httpx.ReadTimeout("response lost", request=request)
        get_count += 1
        assert target_exists
        return httpx.Response(200, json=_mock_response(prepared, created=False))

    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        adapter = MockPortalHTTPAdapter(client)
        first_store = _ExecutionStore(prepared)
        first_registry = create_approved_action_registry(
            adapter=adapter, action_execution_store=first_store
        )
        with collecting_node(sink), pytest.raises(ActionResultUnconfirmedError):
            await first_registry.execute_approved_action(
                _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
            )
        recovery_store = _RecoveryStore(prepared)
        recovery_registry = create_approved_action_registry(
            adapter=adapter, action_execution_store=recovery_store
        )
        with collecting_node(sink):
            await recovery_registry.execute_approved_action(
                _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
            )

    assert post_count == 1
    assert get_count == 1
    assert recovery_store.resend_count == 0

    finishes = [f for f in sink.finishes.values() if f.span_kind == "tool"]
    assert [f.status for f in finishes] == ["failed", "succeeded"]
    assert finishes[0].error_category == "action_result_unconfirmed"
    assert finishes[0].metadata["attempt_number"] == 1
    assert "attempt_number" not in finishes[1].metadata
    assert "retry_count" not in finishes[1].metadata
    assert not [v for v in sink.starts.values() if v[1].span_kind == "action_recovery"]


async def test_crash_after_get_found_repeats_get_only_and_converges() -> None:
    prepared = _prepared()
    get_count = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal get_count
        get_count += 1
        return httpx.Response(200, json=_mock_response(prepared, created=False))

    def crash(point: str) -> None:
        if point == "after_recovery_get_found_before_success":
            raise RuntimeError("injected crash")

    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        store = _RecoveryStore(prepared)
        with pytest.raises(RuntimeError, match="injected crash"):
            await create_approved_action_registry(
                adapter=MockPortalHTTPAdapter(client),
                action_execution_store=store,
                fault_injector=crash,
            ).execute_approved_action(
                _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
            )
        await create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(client), action_execution_store=store
        ).execute_approved_action(
            _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
        )
    assert get_count == 2
    assert store.resend_count == 0


async def test_crash_after_recovery_post_success_restarts_with_get_found() -> None:
    prepared = _prepared()
    target_exists = False
    methods: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal target_exists
        methods.append(request.method)
        if request.method == "GET":
            if not target_exists:
                return httpx.Response(404)
            return httpx.Response(200, json=_mock_response(prepared, created=False))
        target_exists = True
        return httpx.Response(201, json=_mock_response(prepared, created=True))

    def crash(point: str) -> None:
        if point == "after_recovery_http_before_success":
            raise RuntimeError("injected crash")

    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        store = _RecoveryStore(prepared)
        with pytest.raises(RuntimeError, match="injected crash"):
            await create_approved_action_registry(
                adapter=MockPortalHTTPAdapter(client),
                action_execution_store=store,
                fault_injector=crash,
            ).execute_approved_action(
                _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
            )
        await create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(client), action_execution_store=store
        ).execute_approved_action(
            _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
        )
    assert methods == ["GET", "POST", "GET"]
    assert store.resend_count == 1


@pytest.mark.parametrize(
    "case", ["success", "rejected", "unconfirmed", "failed", "outcome_unknown"]
)
async def test_approved_tool_trace_observes_persisted_identity_and_no_retry(case) -> None:
    from app.domain.action_execution import ActionExecutionFailedError
    from app.domain.tracing import current_trace_scope
    from tests.tracing import CollectingTraceSink, collecting_node

    prepared = _prepared()
    prepared = replace(
        prepared, args=prepared.args.model_copy(update={"cover_letter": "ACTION-ARGS-CANARY"})
    )
    if case in {"failed", "outcome_unknown"}:
        prepared = replace(prepared, status=case)
    methods = []

    class Store(_ExecutionStore):
        async def confirm_failure(self, identity, **kwargs):
            self.prepared = replace(self.prepared, status="failed")
            self.failure = kwargs

        async def confirm_success(self, identity, **kwargs):
            await super().confirm_success(identity, **kwargs)
            self.prepared = replace(
                self.prepared,
                status="succeeded",
                result={"external_ref": self.confirmed.external_ref},
            )

    def respond(request):
        methods.append(request.method)
        if case == "rejected":
            return httpx.Response(422)
        if case == "unconfirmed":
            raise httpx.ReadTimeout("EXCEPTION-CANARY", request=request)
        return httpx.Response(201, json=_mock_response(prepared, created=True))

    sink = CollectingTraceSink()
    store = Store(prepared)
    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        registry = create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(client), action_execution_store=store
        )
        with collecting_node(sink) as scope:
            if case == "success":
                result = await registry.execute_approved_action(
                    _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
                )
                assert (
                    await registry.execute_approved_action(
                        _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
                    )
                    == result
                )
            else:
                error = (
                    ActionResultUnconfirmedError
                    if case == "unconfirmed"
                    else ActionOutcomeUnknownError
                    if case == "outcome_unknown"
                    else ActionExecutionFailedError
                )
                with pytest.raises(error):
                    await registry.execute_approved_action(
                        _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
                    )
            assert current_trace_scope() == scope
    observations = [(ctx, span) for ctx, span in sink.starts.values() if span.span_kind == "tool"]
    assert len(observations) == (2 if case == "success" else 1)
    for _ctx, span in observations:
        assert span.parent == scope.parent
        assert dict(span.metadata) == {
            "tool_name": "submit_mock_application",
            "tool_effect": "irreversible",
            "tool_invocation_id": prepared.invocation_id,
        }
    finish = sink.finishes[observations[0][0].context_id]
    assert finish.status == ("succeeded" if case == "success" else "failed")
    if case not in {"failed", "outcome_unknown"}:
        assert finish.metadata["attempt_number"] == 1
        assert finish.metadata["retry_count"] == 0
        assert methods == ["POST"]
    else:
        assert "attempt_number" not in finish.metadata
        assert methods == []
    if case == "success":
        assert finish.metadata["output_bytes"] == len(result.encode("utf-8"))
        duplicate = sink.finishes[observations[1][0].context_id]
        assert duplicate.status == "skipped" and "attempt_number" not in duplicate.metadata
    elif case == "rejected":
        assert store.prepared.status == "failed"
        assert store.failure["error_category"] == "external_rejected_no_effect"
    elif case == "unconfirmed":
        assert store.confirmed is None
        assert finish.error_category == "action_result_unconfirmed"
    for canary in (
        "ACTION-ARGS-CANARY",
        "EXCEPTION-CANARY",
        prepared.idempotency_key,
        "mock-submission:recovered",
        "target_ref",
        "cover_letter",
    ):
        assert canary not in sink.safe_json() + repr(sink.starts) + repr(sink.finishes)


@pytest.mark.parametrize("commit_succeeded", [False, True])
async def test_success_accounting_disconnect_recovers_without_second_post(commit_succeeded):
    from app.domain.errors import DomainUnavailableError

    prepared = _prepared()
    methods = []

    class Store(_RecoveryStore):
        def __init__(self):
            super().__init__(prepared)
            self.prepared = prepared
            self.confirmations = 0

        async def begin_send(self, identity, *, now):
            self.prepared = replace(self.prepared, status="executing")
            return SendAuthorization(execution=self.prepared, allowed=True)

        async def confirm_success(self, identity, **kwargs):
            self.confirmations += 1
            if self.confirmations > 1 or commit_succeeded:
                await super().confirm_success(identity, **kwargs)
                self.prepared = replace(
                    self.prepared,
                    status="succeeded",
                    result={"external_ref": self.confirmed.external_ref},
                )
            if self.confirmations == 1:
                raise DomainUnavailableError(commit_outcome_unknown=True)

    def respond(request):
        methods.append(request.method)
        return httpx.Response(
            201 if request.method == "POST" else 200,
            json=_mock_response(prepared, created=request.method == "POST"),
        )

    store = Store()
    async with httpx.AsyncClient(
        base_url="http://mock.test", transport=httpx.MockTransport(respond)
    ) as client:
        registry = create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(client), action_execution_store=store
        )
        with pytest.raises(ActionResultUnconfirmedError):
            await registry.execute_approved_action(
                _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
            )
        assert methods == ["POST"]
        await registry.execute_approved_action(
            _identity(prepared), deadline=monotonic() + 10, cancellation=_Cancellation()
        )
    assert methods == (["POST"] if commit_succeeded else ["POST", "GET"])
    assert store.resend_count == 0
    assert store.confirmed is not None
