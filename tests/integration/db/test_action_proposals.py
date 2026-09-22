from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from app.db.actions import SqlAlchemyActionStore
from app.db.events import SqlAlchemyRunEventReader
from app.db.models import (
    ActionIntent,
    ApprovalRequest,
    Document,
    Run,
    RunEvent,
    ToolInvocation,
    User,
    WorkspaceMembership,
)
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import AsyncSessionFactory, create_database_engine, create_session_factory
from app.domain.actions import (
    ACTION_KEY,
    PrepareActionCommand,
    SubmitApplicationArgsV1,
    SupersedeActionCommand,
)
from app.domain.errors import (
    DomainConflictError,
    DomainInvariantError,
    DomainNotFoundError,
)
from app.domain.provisioning import ProvisioningService, WorkspaceRole
from app.domain.runs import RunMode, RunService
from app.domain.tenancy import TenantContext
from app.events.contracts import RunEventType
from tests.legacy_runtime import SqlAlchemyRunStore

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)


class _InjectedFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _Runtime:
    sessions: AsyncSessionFactory
    tenant: TenantContext
    run_id: UUID
    resume_document_id: UUID

    def command(
        self,
        *,
        proposal_id: UUID | None = None,
        revision: int = 1,
        cover_letter: str = "Exact synthetic cover letter",
        expires_at: datetime = _NOW + timedelta(hours=1),
    ) -> PrepareActionCommand:
        return PrepareActionCommand(
            tenant=self.tenant,
            run_id=self.run_id,
            action_proposal_id=proposal_id or uuid4(),
            action_key=ACTION_KEY,
            action_revision=revision,
            args=SubmitApplicationArgsV1(
                job_ref="synthetic-job-ref",
                resume_document_id=self.resume_document_id,
                answers={"availability": "Two weeks", "location": "Remote"},
                cover_letter=cover_letter,
            ),
            now=_NOW,
            expires_at=expires_at,
        )


async def _seed_runtime(database_url: str, *, suffix: str) -> tuple[object, _Runtime]:
    engine = create_database_engine(SecretStr(database_url))
    sessions = create_session_factory(engine)
    identity = await ProvisioningService(
        SqlAlchemyProvisioningStore(sessions)
    ).provision_personal_workspace(f"gate6-action-{suffix}")
    tenant = TenantContext(identity.workspace_id, identity.user_id, WorkspaceRole.ADMIN)
    document_id = uuid4()
    content = f"Synthetic immutable resume for {suffix}."
    async with sessions.begin() as session:
        session.add(
            Document(
                id=document_id,
                workspace_id=identity.workspace_id,
                created_by_user_id=identity.user_id,
                title="Synthetic resume",
                source_type="text",
                source_name=f"{suffix}.txt",
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                normalization_version="text-normalization-v1",
                chunking_version="document-chunking-v1",
                embedding_model="qwen-beijing-text-embedding-v4-1536-v1",
            )
        )
    accepted = await RunService(SqlAlchemyRunStore(sessions)).create_run(
        tenant=tenant,
        mode=RunMode.APPLICATION,
        query="Prepare an explicit synthetic application action",
        resume_document_id=document_id,
    )
    async with sessions.begin() as session:
        run = await session.scalar(select(Run).where(Run.id == accepted.run_id).with_for_update())
        assert run is not None
        run.status = "running"
        run.started_at = _NOW
    return engine, _Runtime(sessions, tenant, accepted.run_id, document_id)


async def _counts(runtime: _Runtime) -> tuple[int, int, int, int]:
    async with runtime.sessions() as session:
        return (
            await session.scalar(select(func.count()).select_from(ActionIntent)) or 0,
            await session.scalar(select(func.count()).select_from(ApprovalRequest)) or 0,
            await session.scalar(
                select(func.count())
                .select_from(RunEvent)
                .where(RunEvent.type == RunEventType.ACTION_PROPOSED.value)
            )
            or 0,
            await session.scalar(select(func.count()).select_from(ToolInvocation)) or 0,
        )


async def test_prepare_action_first_commit_and_response_lost_retry_converge_exactly(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="response-lost")
    try:
        command = runtime.command(proposal_id=uuid4())
        store = SqlAlchemyActionStore(runtime.sessions)

        first = await store.prepare_action(command)
        second = await store.prepare_action(command)

        assert second == first
        assert first.intent.action_intent_id == command.action_proposal_id
        assert first.intent.idempotency_key == str(command.action_proposal_id)
        assert first.intent.args_digest == first.approval_request.args_digest
        assert first.intent.target_digest == first.approval_request.target_digest
        assert (
            first.intent.approval_binding_digest == first.approval_request.approval_binding_digest
        )
        assert first.intent.target_snapshot == {
            "provider": "mock_portal",
            "target_type": "internal_mock",
            "target_ref": "default",
            "resource_scope": "application_submission",
        }
        assert first.approval_request.policy_snapshot == {
            "eligible_roles": ["reviewer", "admin"],
            "required_approvals": 1,
            "separation_of_duty": False,
        }
        assert await _counts(runtime) == (1, 1, 1, 0)
    finally:
        await engine.dispose()


async def test_prepare_action_rolls_back_intent_request_event_and_sequence_before_commit(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="prepare-rollback")
    try:
        command = runtime.command(proposal_id=uuid4())
        async with runtime.sessions() as session:
            sequence_before = await session.scalar(
                select(Run.next_event_seq).where(Run.id == runtime.run_id)
            )

        def inject(point: str) -> None:
            if point == "prepare_before_commit":
                raise _InjectedFailure

        with pytest.raises(_InjectedFailure):
            await SqlAlchemyActionStore(
                runtime.sessions,
                fault_injector=inject,  # type: ignore[arg-type]
            ).prepare_action(command)

        assert await _counts(runtime) == (0, 0, 0, 0)
        async with runtime.sessions() as session:
            assert (
                await session.scalar(select(Run.next_event_seq).where(Run.id == runtime.run_id))
                == sequence_before
            )

        prepared = await SqlAlchemyActionStore(runtime.sessions).prepare_action(command)
        assert prepared.intent.action_intent_id == command.action_proposal_id
        assert await _counts(runtime) == (1, 1, 1, 0)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("conflict", ["args", "expiry", "proposal_id"])
async def test_prepare_action_rejects_conflicting_immutable_facts(
    migrated_database_url: str,
    conflict: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix=f"conflict-{conflict}")
    try:
        command = runtime.command(proposal_id=uuid4())
        store = SqlAlchemyActionStore(runtime.sessions)
        await store.prepare_action(command)
        if conflict == "args":
            conflicting = replace(
                command,
                args=command.args.model_copy(update={"cover_letter": "Changed by one byte!"}),
            )
        elif conflict == "expiry":
            conflicting = replace(command, expires_at=command.expires_at + timedelta(seconds=1))
        else:
            conflicting = replace(command, action_proposal_id=uuid4())

        with pytest.raises(DomainInvariantError):
            await store.prepare_action(conflicting)

        assert await _counts(runtime) == (1, 1, 1, 0)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("drift", ["target", "binding"])
async def test_prepare_action_detects_persisted_target_or_binding_drift(
    migrated_database_url: str,
    drift: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix=f"{drift}-drift")
    try:
        command = runtime.command(proposal_id=uuid4())
        store = SqlAlchemyActionStore(runtime.sessions)
        await store.prepare_action(command)
        async with runtime.sessions.begin() as session:
            changed = (
                {"target_snapshot": {"provider": "forged"}}
                if drift == "target"
                else {"approval_binding_digest": f"sha256:{'f' * 64}"}
            )
            await session.execute(
                update(ActionIntent)
                .where(ActionIntent.id == command.action_proposal_id)
                .values(**changed)
            )

        with pytest.raises(DomainInvariantError):
            await store.prepare_action(command)
    finally:
        await engine.dispose()


async def test_database_represents_different_action_keys_at_the_same_revision(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="different-keys")
    try:
        prepared = await SqlAlchemyActionStore(runtime.sessions).prepare_action(
            runtime.command(proposal_id=uuid4())
        )
        second_id = uuid4()
        async with runtime.sessions.begin() as session:
            session.add(
                ActionIntent(
                    id=second_id,
                    workspace_id=runtime.tenant.workspace_id,
                    originating_actor_user_id=runtime.tenant.actor_user_id,
                    run_id=runtime.run_id,
                    action_key="archive_application",
                    action_revision=1,
                    tool_name=prepared.intent.tool_name,
                    effect=prepared.intent.effect.value,
                    args_snapshot=prepared.intent.args_snapshot,
                    canonicalization_version=prepared.intent.canonicalization_version,
                    args_digest=prepared.intent.args_digest,
                    target_snapshot=prepared.intent.target_snapshot,
                    target_canonicalization_version=(
                        prepared.intent.target_canonicalization_version
                    ),
                    target_digest=prepared.intent.target_digest,
                    approval_binding_version=prepared.intent.approval_binding_version,
                    approval_binding_digest=prepared.intent.approval_binding_digest,
                    status="proposed",
                    idempotency_key=str(second_id),
                    recovery_attempts=0,
                )
            )
        async with runtime.sessions() as session:
            identities = set(
                await session.execute(
                    select(ActionIntent.action_key, ActionIntent.action_revision).where(
                        ActionIntent.run_id == runtime.run_id
                    )
                )
            )
        assert identities == {("submit_application", 1), ("archive_application", 1)}
    finally:
        await engine.dispose()


async def test_two_exact_concurrent_prepare_callers_converge_without_sleep(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="concurrent-exact")
    try:
        command = runtime.command(proposal_id=uuid4())
        first_at_commit = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()

        async def block_first(point: str) -> None:
            if point == "prepare_before_commit":
                first_at_commit.set()
                await release_first.wait()

        async def second_caller() -> object:
            second_started.set()
            return await SqlAlchemyActionStore(runtime.sessions).prepare_action(command)

        first_task = asyncio.create_task(
            SqlAlchemyActionStore(
                runtime.sessions,
                fault_injector=block_first,  # type: ignore[arg-type]
            ).prepare_action(command)
        )
        await first_at_commit.wait()
        second_task = asyncio.create_task(second_caller())
        await second_started.wait()
        release_first.set()
        first, second = await asyncio.gather(first_task, second_task)

        assert first == second
        assert await _counts(runtime) == (1, 1, 1, 0)
    finally:
        await engine.dispose()


async def test_two_conflicting_concurrent_prepare_callers_have_one_winner(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="concurrent-conflict")
    try:
        winner = runtime.command(proposal_id=uuid4())
        loser = runtime.command(proposal_id=uuid4(), cover_letter="Conflicting content")
        first_at_commit = asyncio.Event()
        release_first = asyncio.Event()

        async def block_first(point: str) -> None:
            if point == "prepare_before_commit":
                first_at_commit.set()
                await release_first.wait()

        winner_task = asyncio.create_task(
            SqlAlchemyActionStore(
                runtime.sessions,
                fault_injector=block_first,  # type: ignore[arg-type]
            ).prepare_action(winner)
        )
        await first_at_commit.wait()
        loser_task = asyncio.create_task(
            SqlAlchemyActionStore(runtime.sessions).prepare_action(loser)
        )
        release_first.set()
        winner_result = await winner_task
        with pytest.raises(DomainInvariantError):
            await loser_task

        assert winner_result.intent.action_intent_id == winner.action_proposal_id
        assert await _counts(runtime) == (1, 1, 1, 0)
    finally:
        await engine.dispose()


async def test_pending_supersede_is_atomic_idempotent_and_creates_next_revision(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="supersede-pending")
    try:
        store = SqlAlchemyActionStore(runtime.sessions)
        old = runtime.command(proposal_id=uuid4())
        prepared_old = await store.prepare_action(old)
        immutable_old = (
            prepared_old.intent.args_snapshot,
            prepared_old.intent.args_digest,
            prepared_old.intent.target_snapshot,
            prepared_old.intent.target_digest,
            prepared_old.intent.approval_binding_digest,
            prepared_old.approval_request.policy_snapshot,
            prepared_old.approval_request.expires_at,
        )
        new_id = uuid4()
        command = SupersedeActionCommand(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            old_action_intent_id=old.action_proposal_id,
            new_action_proposal_id=new_id,
            action_key=ACTION_KEY,
            action_revision=2,
            args=old.args.model_copy(update={"cover_letter": "Revision two"}),
            now=_NOW,
            expires_at=_NOW + timedelta(hours=2),
        )

        first = await store.supersede_action(command)
        second = await store.supersede_action(command)

        assert first == second
        assert first.intent.action_intent_id == new_id
        assert first.intent.action_revision == 2
        async with runtime.sessions() as session:
            old_intent = await session.get(ActionIntent, old.action_proposal_id)
            old_request = await session.get(
                ApprovalRequest, prepared_old.approval_request.request_id
            )
            event_types = tuple(
                await session.scalars(
                    select(RunEvent.type)
                    .where(RunEvent.run_id == runtime.run_id)
                    .order_by(RunEvent.seq)
                )
            )
        assert old_intent is not None and old_intent.status == "cancelled"
        assert old_request is not None and old_request.status == "expired"
        assert old_request.version == 2
        assert (
            old_intent.args_snapshot,
            old_intent.args_digest,
            old_intent.target_snapshot,
            old_intent.target_digest,
            old_intent.approval_binding_digest,
            old_request.policy_snapshot,
            old_request.expires_at,
        ) == immutable_old
        assert event_types == (
            "run.created",
            "action.proposed",
            "approval.expired",
            "action.proposed",
        )
        assert await _counts(runtime) == (2, 2, 2, 0)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("terminal_status", ["rejected", "expired"])
async def test_terminal_request_supersede_does_not_forge_an_expired_transition(
    migrated_database_url: str,
    terminal_status: str,
) -> None:
    engine, runtime = await _seed_runtime(
        migrated_database_url, suffix=f"terminal-{terminal_status}"
    )
    try:
        store = SqlAlchemyActionStore(runtime.sessions)
        old = runtime.command(proposal_id=uuid4())
        prepared = await store.prepare_action(old)
        async with runtime.sessions.begin() as session:
            request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            assert request is not None
            request.status = terminal_status
            request.version = 2
        command = SupersedeActionCommand(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            old_action_intent_id=old.action_proposal_id,
            new_action_proposal_id=uuid4(),
            action_key=ACTION_KEY,
            action_revision=2,
            args=old.args.model_copy(update={"cover_letter": "Terminal request revision"}),
            now=_NOW,
            expires_at=_NOW + timedelta(hours=2),
        )

        first = await store.supersede_action(command)
        second = await store.supersede_action(command)

        assert first == second
        async with runtime.sessions() as session:
            old_request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            expired_event_count = await session.scalar(
                select(func.count())
                .select_from(RunEvent)
                .where(RunEvent.type == RunEventType.APPROVAL_EXPIRED.value)
            )
        assert old_request is not None and old_request.status == terminal_status
        assert expired_event_count == 0
        assert await _counts(runtime) == (2, 2, 2, 0)
    finally:
        await engine.dispose()


async def test_approved_unconsumed_request_can_be_superseded_but_consumed_cannot(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="supersede-status")
    try:
        store = SqlAlchemyActionStore(runtime.sessions)
        old = runtime.command(proposal_id=uuid4())
        prepared = await store.prepare_action(old)
        async with runtime.sessions.begin() as session:
            request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            assert request is not None
            request.status = "approved"
            request.version = 2
        command = SupersedeActionCommand(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            old_action_intent_id=old.action_proposal_id,
            new_action_proposal_id=uuid4(),
            action_key=ACTION_KEY,
            action_revision=2,
            args=old.args.model_copy(update={"cover_letter": "Approved revision"}),
            now=_NOW,
            expires_at=_NOW + timedelta(hours=2),
        )
        revised = await store.supersede_action(command)
        assert revised.intent.action_revision == 2

        async with runtime.sessions.begin() as session:
            new_request = await session.get(ApprovalRequest, revised.approval_request.request_id)
            assert new_request is not None
            new_request.status = "consumed"
            new_request.version = 2
            new_request.consumed_at = new_request.created_at
        consumed_command = replace(
            command,
            old_action_intent_id=revised.intent.action_intent_id,
            new_action_proposal_id=uuid4(),
            action_revision=3,
        )
        with pytest.raises(DomainConflictError):
            await store.supersede_action(consumed_command)
        assert await _counts(runtime) == (2, 2, 2, 0)
    finally:
        await engine.dispose()


async def test_supersede_rollback_restores_old_aggregate_event_sequence_and_new_absence(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="supersede-rollback")
    try:
        old = runtime.command(proposal_id=uuid4())
        prepared = await SqlAlchemyActionStore(runtime.sessions).prepare_action(old)
        command = SupersedeActionCommand(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            old_action_intent_id=old.action_proposal_id,
            new_action_proposal_id=uuid4(),
            action_key=ACTION_KEY,
            action_revision=2,
            args=old.args.model_copy(update={"cover_letter": "Rollback revision"}),
            now=_NOW,
            expires_at=_NOW + timedelta(hours=2),
        )
        async with runtime.sessions() as session:
            sequence_before = await session.scalar(
                select(Run.next_event_seq).where(Run.id == runtime.run_id)
            )

        def inject(point: str) -> None:
            if point == "supersede_after_expiry_before_new_proposal":
                raise _InjectedFailure

        with pytest.raises(_InjectedFailure):
            await SqlAlchemyActionStore(
                runtime.sessions,
                fault_injector=inject,  # type: ignore[arg-type]
            ).supersede_action(command)

        async with runtime.sessions() as session:
            old_intent = await session.get(ActionIntent, old.action_proposal_id)
            old_request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            sequence_after = await session.scalar(
                select(Run.next_event_seq).where(Run.id == runtime.run_id)
            )
            expired_events = await session.scalar(
                select(func.count())
                .select_from(RunEvent)
                .where(RunEvent.type == RunEventType.APPROVAL_EXPIRED.value)
            )
        assert old_intent is not None and old_intent.status == "proposed"
        assert old_request is not None and old_request.status == "pending"
        assert sequence_after == sequence_before
        assert expired_events == 0
        assert await _counts(runtime) == (1, 1, 1, 0)

        converged = await SqlAlchemyActionStore(runtime.sessions).supersede_action(command)
        assert converged.intent.action_revision == 2
        assert await _counts(runtime) == (2, 2, 2, 0)
    finally:
        await engine.dispose()


async def test_tenant_run_actor_and_resume_splices_fail_closed(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="tenant-splice")
    second_engine = None
    try:
        store = SqlAlchemyActionStore(runtime.sessions)
        command = runtime.command(proposal_id=uuid4())
        await store.prepare_action(command)

        second_accepted = await RunService(SqlAlchemyRunStore(runtime.sessions)).create_run(
            tenant=runtime.tenant,
            mode=RunMode.APPLICATION,
            query="Second application run",
            resume_document_id=runtime.resume_document_id,
        )
        async with runtime.sessions.begin() as session:
            second_run = await session.scalar(
                select(Run).where(Run.id == second_accepted.run_id).with_for_update()
            )
            assert second_run is not None
            second_run.status = "running"
            second_run.started_at = _NOW

        foreign_user_id = uuid4()
        foreign_subject = "gate6-foreign-actor"
        async with runtime.sessions.begin() as session:
            session.add(User(id=foreign_user_id, auth_subject=foreign_subject))
            session.add(
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=runtime.tenant.workspace_id,
                    user_id=foreign_user_id,
                    role="member",
                )
            )
        foreign_tenant = TenantContext(
            runtime.tenant.workspace_id,
            foreign_user_id,
            WorkspaceRole.MEMBER,
        )
        with pytest.raises(DomainNotFoundError):
            await store.prepare_action(replace(command, tenant=foreign_tenant))
        with pytest.raises(DomainNotFoundError):
            await store.prepare_action(replace(command, run_id=uuid4()))
        with pytest.raises(DomainInvariantError):
            await store.prepare_action(replace(command, run_id=second_accepted.run_id))
        with pytest.raises(DomainInvariantError):
            await store.prepare_action(
                replace(
                    command,
                    args=command.args.model_copy(update={"resume_document_id": uuid4()}),
                )
            )
        second_engine, second_runtime = await _seed_runtime(
            migrated_database_url,
            suffix="other-workspace",
        )
        with pytest.raises(DomainNotFoundError):
            await store.prepare_action(replace(command, tenant=second_runtime.tenant))
        assert await _counts(runtime) == (1, 1, 1, 0)
    finally:
        if second_engine is not None:
            await second_engine.dispose()
        await engine.dispose()


async def test_action_events_use_generic_reader_after_seq_and_redact_business_body(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="event-reader")
    try:
        canary = "cover-letter-secret-canary"
        command = runtime.command(proposal_id=uuid4(), cover_letter=canary)
        store = SqlAlchemyActionStore(runtime.sessions)
        await store.prepare_action(command)

        batch = await SqlAlchemyRunEventReader(runtime.sessions).read_after(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            after_seq=1,
            limit=10,
        )

        assert [event.type for event in batch.events] == [RunEventType.ACTION_PROPOSED]
        assert batch.events[0].seq == 2
        assert batch.high_watermark == 2
        assert canary not in str(batch.events[0].payload)
        assert "answers" not in batch.events[0].payload
        assert "cover_letter" not in batch.events[0].payload
        await store.supersede_action(
            SupersedeActionCommand(
                tenant=runtime.tenant,
                run_id=runtime.run_id,
                old_action_intent_id=command.action_proposal_id,
                new_action_proposal_id=uuid4(),
                action_key=ACTION_KEY,
                action_revision=2,
                args=command.args.model_copy(update={"cover_letter": "safe revision"}),
                now=_NOW,
                expires_at=_NOW + timedelta(hours=2),
            )
        )
        resumed_batch = await SqlAlchemyRunEventReader(runtime.sessions).read_after(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            after_seq=2,
            limit=10,
        )
        assert [event.type for event in resumed_batch.events] == [
            RunEventType.APPROVAL_EXPIRED,
            RunEventType.ACTION_PROPOSED,
        ]
        assert [event.seq for event in resumed_batch.events] == [3, 4]
        assert resumed_batch.high_watermark == 4
        assert canary not in str(tuple(event.payload for event in resumed_batch.events))
        assert await _counts(runtime) == (2, 2, 2, 0)
    finally:
        await engine.dispose()


async def test_approval_request_same_run_composite_fk_rejects_spliced_action(
    migrated_database_url: str,
) -> None:
    engine, runtime = await _seed_runtime(migrated_database_url, suffix="approval-fk")
    try:
        prepared = await SqlAlchemyActionStore(runtime.sessions).prepare_action(
            runtime.command(proposal_id=uuid4())
        )
        async with runtime.sessions.begin() as session:
            original = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            assert original is not None
            session.add(
                ApprovalRequest(
                    id=uuid4(),
                    workspace_id=runtime.tenant.workspace_id,
                    run_id=uuid4(),
                    action_intent_id=prepared.intent.action_intent_id,
                    status="pending",
                    args_digest=original.args_digest,
                    target_digest=original.target_digest,
                    approval_binding_version=original.approval_binding_version,
                    approval_binding_digest=original.approval_binding_digest,
                    policy_version=original.policy_version,
                    policy_snapshot=original.policy_snapshot,
                    version=1,
                    expires_at=original.expires_at,
                )
            )
            with pytest.raises(IntegrityError):
                await session.flush()
    finally:
        await engine.dispose()
