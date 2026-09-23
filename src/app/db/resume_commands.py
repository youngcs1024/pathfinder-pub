"""Atomic acceptance and replay of resume business commands.

Concrete writers are installed only when their business feature has a handler.
They use the supplied transaction for authorization, intent, and Run/Job writes.
"""

from typing import Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ResumeCommand, Run, RunEvent, RunJob, WorkspaceMembership
from app.db.session import AsyncSessionFactory, transaction
from app.domain.errors import DomainConflictError, DomainInvariantError, DomainNotFoundError
from app.domain.resume_commands import (
    CommandAccepted,
    CommandReceiptV1,
    ResumeCommandRequest,
    SessionWriteState,
    require_current_session_write,
)
from app.domain.tenancy import TenantContext

_UNIQUE_REQUEST = "uq_resume_commands_actor_request"


class ResumeCommandWriter(Protocol):
    supported_kinds: frozenset[str]

    async def authorize_and_lock(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        request: ResumeCommandRequest,
    ) -> SessionWriteState | None:
        """Check current resource access and lock a session row if applicable."""
        ...

    async def apply(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        request: ResumeCommandRequest,
        command_id: UUID,
    ) -> CommandReceiptV1:
        """Write the specific intent and any Run, Job, or event in this transaction."""
        ...


async def _require_current_tenant(session: AsyncSession, tenant: TenantContext) -> None:
    role = await session.scalar(
        select(WorkspaceMembership.role)
        .where(
            WorkspaceMembership.workspace_id == tenant.workspace_id,
            WorkspaceMembership.user_id == tenant.actor_user_id,
            WorkspaceMembership.revoked_at.is_(None),
            WorkspaceMembership.role == tenant.role.value,
        )
        .with_for_update(read=True)
    )
    if role is None:
        raise DomainNotFoundError


def _accepted(row: ResumeCommand, request: ResumeCommandRequest) -> CommandAccepted:
    if row.digest_version != 1 or row.receipt_version != 1:
        raise DomainInvariantError("persisted resume command version is unsupported")
    if row.request_digest != request.digest_v1():
        raise DomainConflictError("resume command conflicts with accepted request")
    try:
        receipt = CommandReceiptV1.model_validate(row.receipt_json)
    except ValidationError:
        raise DomainInvariantError("persisted resume command receipt is invalid") from None
    if (
        receipt.command_id != row.id
        or receipt.run_id != row.run_id
        or (receipt.status == "queued") != (receipt.run_id is not None)
    ):
        raise DomainInvariantError("persisted resume command receipt is inconsistent")
    return CommandAccepted(receipt=receipt, replayed=True)


class SqlAlchemyResumeCommandStore:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def accept(
        self,
        *,
        tenant: TenantContext,
        request: ResumeCommandRequest,
        writer: ResumeCommandWriter,
    ) -> CommandAccepted:
        if request.kind not in writer.supported_kinds:
            raise DomainInvariantError("resume command handler is not registered for this kind")
        digest = request.digest_v1()
        command_id = uuid4()
        try:
            async with transaction(self._session_factory) as session:
                await _require_current_tenant(session, tenant)
                state = await writer.authorize_and_lock(session, tenant, request)
                existing = await self._find(session, tenant, request.client_request_id)
                if existing is not None:
                    return _accepted(existing, request)
                if request.session_write is not None:
                    if state is None:
                        raise DomainInvariantError("session command writer did not return state")
                    require_current_session_write(request.session_write, state)
                receipt = await writer.apply(session, tenant, request, command_id)
                if receipt.command_id != command_id:
                    raise DomainInvariantError("command writer returned a different command")
                if (receipt.status == "queued") != (receipt.run_id is not None):
                    raise DomainInvariantError("command writer returned an invalid Run receipt")
                await session.flush()
                if receipt.run_id is not None:
                    run_status = await session.scalar(
                        select(Run.status).where(
                            Run.workspace_id == tenant.workspace_id,
                            Run.id == receipt.run_id,
                            Run.created_by_user_id == tenant.actor_user_id,
                        )
                    )
                    job_status = await session.scalar(
                        select(RunJob.status).where(
                            RunJob.workspace_id == tenant.workspace_id,
                            RunJob.run_id == receipt.run_id,
                            RunJob.originating_actor_user_id == tenant.actor_user_id,
                        )
                    )
                    event_id = await session.scalar(
                        select(RunEvent.id).where(
                            RunEvent.workspace_id == tenant.workspace_id,
                            RunEvent.run_id == receipt.run_id,
                            RunEvent.seq == 1,
                            RunEvent.type == "run.created",
                        )
                    )
                    if run_status != "queued" or job_status != "queued" or event_id is None:
                        raise DomainInvariantError("queued command did not enqueue a valid Run")
                session.add(
                    ResumeCommand(
                        id=command_id,
                        workspace_id=tenant.workspace_id,
                        actor_user_id=tenant.actor_user_id,
                        client_request_id=request.client_request_id,
                        kind=request.kind,
                        digest_version=1,
                        request_digest=digest,
                        receipt_version=1,
                        receipt_json=receipt.model_dump(mode="json"),
                        run_id=receipt.run_id,
                    )
                )
                await session.flush()
            return CommandAccepted(receipt=receipt)
        except IntegrityError as error:
            if (
                getattr(error.orig, "sqlstate", None) != "23505"
                or getattr(getattr(error.orig, "diag", None), "constraint_name", None)
                != _UNIQUE_REQUEST
            ):
                raise
        # The losing transaction is closed. A fresh transaction checks current access
        # before reading the winning receipt, as it does for an ordinary replay.
        async with transaction(self._session_factory) as session:
            await _require_current_tenant(session, tenant)
            await writer.authorize_and_lock(session, tenant, request)
            existing = await self._find(session, tenant, request.client_request_id)
            if existing is None:
                raise DomainInvariantError("resume command conflict winner is missing")
            return _accepted(existing, request)

    @staticmethod
    async def _find(
        session: AsyncSession, tenant: TenantContext, client_request_id: UUID
    ) -> ResumeCommand | None:
        return await session.scalar(
            select(ResumeCommand).where(
                ResumeCommand.workspace_id == tenant.workspace_id,
                ResumeCommand.actor_user_id == tenant.actor_user_id,
                ResumeCommand.client_request_id == client_request_id,
            )
        )
