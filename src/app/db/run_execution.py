from __future__ import annotations

import json
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select

from app.db.models import Document, Run, RunJob, WorkspaceMembership
from app.db.session import AsyncSessionFactory, database_session
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchRequestV1
from app.domain.run_execution import (
    RunExecutionCancelledError,
    RunExecutionInput,
    RunExecutionInvalidError,
    RunExecutionLimitsV1,
)
from app.domain.runs import EXECUTABLE_GRAPH_VERSIONS, RunMode, RunStatus

_NONTERMINAL_RUN_STATUSES = (
    RunStatus.QUEUED.value,
    RunStatus.RUNNING.value,
    RunStatus.WAITING_APPROVAL.value,
)


class SqlAlchemyRunExecutionReader:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def read_for_execution(
        self,
        *,
        run_id: UUID,
        workspace_id: UUID,
        actor_user_id: UUID,
        graph_version: str,
    ) -> RunExecutionInput:
        if not all(isinstance(value, UUID) for value in (run_id, workspace_id, actor_user_id)):
            raise RunExecutionInvalidError("invalid_run_identity")
        if graph_version not in EXECUTABLE_GRAPH_VERSIONS:
            raise RunExecutionInvalidError("unknown_graph_version")
        async with database_session(self._session_factory) as session:
            membership = (
                await session.execute(
                    select(
                        WorkspaceMembership.id,
                        WorkspaceMembership.revoked_at,
                        WorkspaceMembership.role,
                    ).where(
                        WorkspaceMembership.workspace_id == workspace_id,
                        WorkspaceMembership.user_id == actor_user_id,
                    )
                )
            ).one_or_none()
            if membership is None:
                raise RunExecutionInvalidError("run_actor_mismatch")
            if membership.revoked_at is not None:
                raise RunExecutionCancelledError
            row = (
                await session.execute(
                    select(
                        Run.id,
                        Run.workspace_id,
                        Run.created_by_user_id,
                        Run.conversation_id,
                        Run.graph_version,
                        Run.mode,
                        Run.resume_document_id,
                        Run.status,
                        Run.cancel_requested_at,
                        Run.input_json,
                        Run.limits_json,
                        RunJob.resume_approval_request_id,
                    ).where(
                        Run.id == run_id,
                        Run.workspace_id == workspace_id,
                        RunJob.workspace_id == Run.workspace_id,
                        RunJob.run_id == Run.id,
                    )
                )
            ).one_or_none()

        if row is None:
            raise RunExecutionInvalidError("run_not_found")
        if row.created_by_user_id != actor_user_id:
            raise RunExecutionInvalidError("run_actor_mismatch")
        if row.graph_version not in EXECUTABLE_GRAPH_VERSIONS or row.graph_version != graph_version:
            raise RunExecutionInvalidError("unknown_graph_version")
        try:
            mode = RunMode(row.mode)
        except ValueError:
            raise RunExecutionInvalidError("invalid_run_input") from None
        if mode is RunMode.APPLICATION and row.resume_document_id is None:
            raise RunExecutionInvalidError("invalid_run_input")
        if row.cancel_requested_at is not None or row.status == RunStatus.CANCELLED.value:
            raise RunExecutionCancelledError
        if row.status != RunStatus.RUNNING.value:
            raise RunExecutionInvalidError("invalid_run_state")

        try:
            request = ResearchRequestV1.model_validate_json(
                json.dumps(row.input_json, allow_nan=False, separators=(",", ":")),
                strict=True,
            )
            limits = RunExecutionLimitsV1.model_validate_json(
                json.dumps(row.limits_json, allow_nan=False, separators=(",", ":")),
                strict=True,
            )
        except (TypeError, ValidationError, ValueError):
            raise RunExecutionInvalidError("invalid_run_input") from None
        if request.include_application_draft is not (mode is RunMode.APPLICATION):
            raise RunExecutionInvalidError("invalid_run_input")
        if row.resume_document_id is not None:
            async with database_session(self._session_factory) as session:
                bound_document = await session.scalar(
                    select(Document.id).where(
                        Document.workspace_id == row.workspace_id,
                        Document.id == row.resume_document_id,
                    )
                )
            if bound_document is None:
                raise RunExecutionInvalidError("invalid_run_input")
        return RunExecutionInput(
            run_id=row.id,
            workspace_id=row.workspace_id,
            actor_user_id=row.created_by_user_id,
            conversation_id=row.conversation_id,
            graph_version=row.graph_version,
            mode=mode,
            resume_document_id=row.resume_document_id,
            request=request,
            limits=limits,
            role=WorkspaceRole(membership.role),
            resume_approval_request_id=row.resume_approval_request_id,
        )

    async def assert_execution_allowed(
        self,
        *,
        run_id: UUID,
        workspace_id: UUID,
        actor_user_id: UUID,
        graph_version: str,
    ) -> None:
        if not all(isinstance(value, UUID) for value in (run_id, workspace_id, actor_user_id)):
            raise RunExecutionInvalidError("invalid_run_identity")
        if graph_version not in EXECUTABLE_GRAPH_VERSIONS:
            raise RunExecutionInvalidError("unknown_graph_version")
        async with database_session(self._session_factory) as session:
            membership = (
                await session.execute(
                    select(
                        WorkspaceMembership.id,
                        WorkspaceMembership.revoked_at,
                    ).where(
                        WorkspaceMembership.workspace_id == workspace_id,
                        WorkspaceMembership.user_id == actor_user_id,
                    )
                )
            ).one_or_none()
            if membership is None:
                raise RunExecutionInvalidError("run_actor_mismatch")
            if membership.revoked_at is not None:
                raise RunExecutionCancelledError
            row = (
                await session.execute(
                    select(
                        Run.created_by_user_id,
                        Run.graph_version,
                        Run.status,
                        Run.cancel_requested_at,
                    ).where(
                        Run.id == run_id,
                        Run.workspace_id == workspace_id,
                    )
                )
            ).one_or_none()

        if row is None:
            raise RunExecutionInvalidError("run_not_found")
        if row.created_by_user_id != actor_user_id:
            raise RunExecutionInvalidError("run_actor_mismatch")
        if row.graph_version not in EXECUTABLE_GRAPH_VERSIONS or row.graph_version != graph_version:
            raise RunExecutionInvalidError("unknown_graph_version")
        if row.cancel_requested_at is not None or row.status == RunStatus.CANCELLED.value:
            raise RunExecutionCancelledError
        if row.status != RunStatus.RUNNING.value:
            raise RunExecutionInvalidError("invalid_run_state")

    async def has_nonterminal_other_graph_versions(self, graph_version: str) -> bool:
        async with database_session(self._session_factory) as session:
            run_id = await session.scalar(
                select(Run.id)
                .where(
                    Run.status.in_(_NONTERMINAL_RUN_STATUSES),
                    Run.graph_version != graph_version,
                )
                .limit(1)
            )
        return run_id is not None
