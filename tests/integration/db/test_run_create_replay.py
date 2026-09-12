from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError

import app.db.runs as runs_module
from app.db.models import Run, RunEvent, WorkspaceMembership
from app.db.runs import SqlAlchemyRunStore
from app.db.session import transaction
from app.domain.errors import (
    DomainConflictError,
    DomainInvariantError,
    DomainNotFoundError,
    DomainUnavailableError,
    DomainValidationError,
)
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchRequestV1
from app.domain.runs import (
    CURRENT_GRAPH_VERSION,
    DEFAULT_RUN_LIMITS,
    RunAccepted,
    RunCreateIdentity,
    RunMode,
    RunStatus,
    create_request_digest_v1,
)
from app.domain.tenancy import TenantContext
from tests.integration.db import test_run_api_store as support
from tests.integration.db.test_run_api_store import runtime as runtime

pytestmark = pytest.mark.integration
QUERY = "Synthetic private query canary E33"


async def _create(
    store: SqlAlchemyRunStore,
    tenant: TenantContext,
    key: UUID | None,
    *,
    query: str = QUERY,
    **overrides: object,
) -> RunAccepted:
    arguments = {
        "tenant": tenant,
        "mode": RunMode.RESEARCH,
        "resume_document_id": None,
        "request": ResearchRequestV1(query=query),
        "limits": dict(DEFAULT_RUN_LIMITS),
        "graph_version": CURRENT_GRAPH_VERSION,
        "request_identity": (
            RunCreateIdentity(key, create_request_digest_v1(mode=RunMode.RESEARCH, query=query))
            if key is not None
            else None
        ),
        **overrides,
    }
    return await store.create_run(**arguments)


@pytest.mark.parametrize("status", ["queued", "completed", "failed", "cancelled"])
async def test_replay_returns_receipt_without_changing_persisted_state(
    runtime: support._Runtime, status: str
) -> None:
    tenant = await support._personal_tenant(runtime, "e33-replay")
    store = SqlAlchemyRunStore(runtime.session_factory)
    key = uuid4()
    first = await _create(store, tenant, key)
    assert first.replayed is False
    async with transaction(runtime.session_factory) as session:
        now = datetime.now(UTC)
        await session.execute(
            update(Run)
            .where(Run.id == first.run_id)
            .values(
                status=status,
                started_at=now if status in {"completed", "failed"} else None,
                finished_at=now if status != "queued" else None,
                result_json={} if status == "completed" else None,
                error_category="synthetic_failure" if status == "failed" else None,
            )
        )
    before = await support._business_counts(runtime.session_factory)
    replay = await _create(store, tenant, key, limits={"max_model_calls": 1})
    assert replay == RunAccepted(first.run_id, RunStatus.QUEUED, replayed=True)
    assert await support._business_counts(runtime.session_factory) == before == (1, 1, 1, 1, 1)
    async with runtime.session_factory() as session:
        run = await session.get(Run, first.run_id)
        assert run.status == status
        assert run.limits_json == DEFAULT_RUN_LIMITS
        assert run.client_request_id == key
        assert run.create_request_version == 1
        assert run.next_event_seq == 2
        event = (await session.scalars(select(RunEvent))).one()
        assert QUERY not in str(event.payload)
        assert str(key) not in str(event.payload)
        assert run.create_request_digest not in str(event.payload)


async def test_legacy_and_request_scope_are_independent(runtime: support._Runtime) -> None:
    first = await support._personal_tenant(runtime, "e33-scope-first")
    second = await support._personal_tenant(runtime, "e33-scope-second")
    async with transaction(runtime.session_factory) as session:
        session.add(
            WorkspaceMembership(
                workspace_id=first.workspace_id,
                user_id=second.actor_user_id,
                role="member",
            )
        )
        session.add(
            WorkspaceMembership(
                workspace_id=second.workspace_id,
                user_id=first.actor_user_id,
                role="member",
            )
        )
    same_workspace = TenantContext(first.workspace_id, second.actor_user_id, WorkspaceRole.MEMBER)
    other_workspace = TenantContext(second.workspace_id, first.actor_user_id, WorkspaceRole.MEMBER)
    store = SqlAlchemyRunStore(runtime.session_factory)
    key = uuid4()
    receipts = [
        await _create(store, tenant, request_key)
        for tenant, request_key in (
            (first, key),
            (same_workspace, key),
            (other_workspace, key),
            (first, None),
            (first, None),
        )
    ]
    assert len({receipt.run_id for receipt in receipts}) == 5
    assert all(not receipt.replayed for receipt in receipts)
    assert await support._business_counts(runtime.session_factory) == (5, 5, 5, 5, 5)


async def test_content_conflict_and_invalid_identity_do_not_write(
    runtime: support._Runtime,
) -> None:
    tenant = await support._personal_tenant(runtime, "e33-conflict")
    store = SqlAlchemyRunStore(runtime.session_factory)
    key = uuid4()
    await _create(store, tenant, key)
    with pytest.raises(DomainConflictError) as error:
        await _create(store, tenant, key, query="Different synthetic query")
    assert str(error.value) == "run creation request conflicts with accepted request"
    with pytest.raises(DomainValidationError):
        await _create(store, tenant, uuid4(), request_identity=RunCreateIdentity(uuid4(), "0" * 64))
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)


@pytest.mark.parametrize("revoke", [True, False])
async def test_replay_rechecks_current_membership_and_role(
    runtime: support._Runtime, revoke: bool
) -> None:
    tenant = await support._personal_tenant(runtime, "e33-auth")
    store = SqlAlchemyRunStore(runtime.session_factory)
    key = uuid4()
    await _create(store, tenant, key)
    values = {"revoked_at": datetime.now(UTC)} if revoke else {"role": "member"}
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(WorkspaceMembership.workspace_id == tenant.workspace_id)
            .values(**values)
        )
    with pytest.raises(DomainNotFoundError):
        await _create(store, tenant, key)
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)


@pytest.mark.parametrize("different", [False, True])
async def test_two_creators_race_and_loser_rolls_back_every_table(
    runtime: support._Runtime, different: bool
) -> None:
    tenant = await support._personal_tenant(runtime, "e33-race")
    barrier = asyncio.Barrier(2)
    sessions_seen = []

    class RacingStore(SqlAlchemyRunStore):
        async def _find_request(self, session, tenant, identity):
            sessions_seen.append(session)
            found = await super()._find_request(session, tenant, identity)
            if found is None:
                await barrier.wait()
            return found

    store = RacingStore(runtime.session_factory)
    key = uuid4()
    async with asyncio.timeout(15):
        results = await asyncio.gather(
            _create(store, tenant, key),
            _create(store, tenant, key, query="Different synthetic query" if different else QUERY),
            return_exceptions=True,
        )
    receipts = [result for result in results if isinstance(result, RunAccepted)]
    if different:
        assert len(receipts) == 1
        assert sum(isinstance(result, DomainConflictError) for result in results) == 1
    else:
        assert len(receipts) == 2
        assert receipts[0].run_id == receipts[1].run_id
        assert sorted(receipt.replayed for receipt in receipts) == [False, True]
    assert len(sessions_seen) == 3
    assert len({id(session) for session in sessions_seen}) == 3
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)


@pytest.mark.parametrize("outcome", ["commit", "rollback", "lock_timeout"])
async def test_uncommitted_winner_controls_unique_wait_result(
    runtime: support._Runtime, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    tenant = await support._personal_tenant(runtime, "e33-unique-wait")
    store = SqlAlchemyRunStore(runtime.session_factory)
    key = uuid4()
    ready = asyncio.Event()
    release = asyncio.Event()
    loser_pid_ready = asyncio.Event()
    loser_pid = None
    winner_task = None
    loser_task = None

    class RollbackWinner(Exception):
        pass

    @asynccontextmanager
    async def controlled_transaction(factory):
        nonlocal loser_pid
        async with transaction(factory) as session:
            if asyncio.current_task() is loser_task:
                await session.execute(text("SET LOCAL lock_timeout = '3000ms'"))
                loser_pid = await session.scalar(text("SELECT pg_backend_pid()"))
                loser_pid_ready.set()
            yield session
            if asyncio.current_task() is winner_task:
                ready.set()
                await release.wait()
                if outcome == "rollback":
                    raise RollbackWinner

    monkeypatch.setattr(runs_module, "transaction", controlled_transaction)
    winner_task = asyncio.create_task(_create(store, tenant, key))
    try:
        async with asyncio.timeout(15):
            await ready.wait()
            loser_task = asyncio.create_task(_create(store, tenant, key))
            await loser_pid_ready.wait()
            # Observe the actual unique-index wait before releasing the winner.
            async with runtime.session_factory() as observer:
                while not await observer.scalar(
                    text("SELECT cardinality(pg_blocking_pids(:pid)) > 0"), {"pid": loser_pid}
                ):
                    if loser_task.done():
                        pytest.fail("loser did not reach the expected unique lock wait")
                    await asyncio.sleep(0.01)
            if outcome == "lock_timeout":
                with pytest.raises(DomainUnavailableError) as error:
                    await loser_task
                assert error.value.commit_outcome_unknown is False
            release.set()
            if outcome == "rollback":
                with pytest.raises(RollbackWinner):
                    await winner_task
                receipt = await loser_task
                assert receipt.replayed is False
            else:
                winner = await winner_task
                if outcome == "commit":
                    loser = await loser_task
                    assert loser == RunAccepted(winner.run_id, RunStatus.QUEUED, replayed=True)
    finally:
        release.set()
        tasks = [task for task in (winner_task, loser_task) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)


@pytest.mark.parametrize("failure", ["missing", "revoked"])
async def test_conflict_recovery_is_bounded_and_reauthorizes(
    runtime: support._Runtime, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    tenant = await support._personal_tenant(runtime, "e33-recovery")
    key = uuid4()
    await _create(SqlAlchemyRunStore(runtime.session_factory), tenant, key)
    lookups = 0

    class MissedWinnerStore(SqlAlchemyRunStore):
        async def _find_request(self, session, tenant, identity):
            nonlocal lookups
            lookups += 1
            return None

    @asynccontextmanager
    async def revoke_after_rollback(factory):
        try:
            async with transaction(factory) as session:
                yield session
        except IntegrityError:
            if failure == "revoked":
                async with transaction(factory) as session:
                    await session.execute(
                        update(WorkspaceMembership)
                        .where(WorkspaceMembership.workspace_id == tenant.workspace_id)
                        .values(revoked_at=datetime.now(UTC))
                    )
            raise

    monkeypatch.setattr(runs_module, "transaction", revoke_after_rollback)
    expected = DomainInvariantError if failure == "missing" else DomainNotFoundError
    with pytest.raises(expected) as error:
        await _create(MissedWinnerStore(runtime.session_factory), tenant, key)
    assert lookups == (2 if failure == "missing" else 1)
    assert error.value.__context__ is None
    if failure == "missing":
        assert str(error.value) == "run creation conflict winner is missing"
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)


async def test_unrelated_integrity_failure_is_not_replay(
    runtime: support._Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    tenant = await support._personal_tenant(runtime, "e33-other-constraint")
    store = SqlAlchemyRunStore(runtime.session_factory)
    first = await _create(store, tenant, uuid4())
    async with runtime.session_factory() as session:
        conversation_id = await session.scalar(
            select(Run.conversation_id).where(Run.id == first.run_id)
        )
    monkeypatch.setattr(runs_module, "uuid4", lambda: conversation_id)
    with pytest.raises(IntegrityError) as error:
        await _create(store, tenant, uuid4())
    assert error.value.orig.sqlstate == "23505"
    assert error.value.orig.diag.constraint_name != "uq_runs_workspace_creator_request"
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)


def test_unknown_persisted_version_fails_closed() -> None:
    run = Run(id=uuid4(), create_request_version=2, create_request_digest="0" * 64)
    with pytest.raises(DomainInvariantError, match="identity version is unsupported"):
        SqlAlchemyRunStore._replay(run, RunMode.RESEARCH, ResearchRequestV1(query=QUERY), None)


@pytest.mark.parametrize("committed", [False, True])
async def test_connection_failure_does_not_trigger_implicit_replay(
    runtime: support._Runtime, monkeypatch: pytest.MonkeyPatch, committed: bool
) -> None:
    tenant = await support._personal_tenant(runtime, "e33-unavailable")
    store = SqlAlchemyRunStore(runtime.session_factory)
    key = uuid4()
    attempts = 0

    @asynccontextmanager
    async def unavailable_transaction(factory):
        nonlocal attempts
        attempts += 1
        if not committed:
            raise DomainUnavailableError
        async with transaction(factory) as session:
            yield session
        # Simulate losing the acknowledgement after a real successful commit.
        raise DomainUnavailableError(commit_outcome_unknown=True)

    monkeypatch.setattr(runs_module, "transaction", unavailable_transaction)
    with pytest.raises(DomainUnavailableError) as error:
        await _create(store, tenant, key)
    assert error.value.commit_outcome_unknown is committed
    assert attempts == 1
    expected = 1 if committed else 0
    assert await support._business_counts(runtime.session_factory) == (expected,) * 5
    monkeypatch.setattr(runs_module, "transaction", transaction)
    accepted = await _create(store, tenant, key)
    assert accepted.replayed is committed
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)


async def test_keyed_application_preserves_document_scope_and_content_binding(
    runtime: support._Runtime,
) -> None:
    tenant = await support._personal_tenant(runtime, "e33-application")
    other = await support._personal_tenant(runtime, "e33-other-document")
    resume = await support._resume(runtime, tenant)
    foreign_resume = await support._resume(runtime, other, suffix="b")
    store = SqlAlchemyRunStore(runtime.session_factory)
    key = uuid4()

    async def application(document_id: UUID):
        return await _create(
            store,
            tenant,
            key,
            mode=RunMode.APPLICATION,
            resume_document_id=document_id,
            request=ResearchRequestV1(query=QUERY, include_application_draft=True),
            request_identity=RunCreateIdentity(
                key,
                create_request_digest_v1(
                    mode=RunMode.APPLICATION, query=QUERY, resume_document_id=document_id
                ),
            ),
        )

    with pytest.raises(DomainNotFoundError):
        await application(foreign_resume)
    assert await support._business_counts(runtime.session_factory) == (0, 0, 0, 0, 0)
    first = await application(resume)
    assert (await application(resume)).run_id == first.run_id
    with pytest.raises(DomainConflictError):
        await application(foreign_resume)
    with pytest.raises(DomainConflictError):
        await _create(store, tenant, key)
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)
