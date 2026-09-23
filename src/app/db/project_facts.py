"""Immutable project fact versions and authorized review commands."""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    MaterialFact,
    MaterialFactEvidence,
    MaterialFactSet,
    MaterialFactVersion,
    MaterialImport,
    MaterialProject,
    MaterialSnapshot,
    MaterialSnapshotFile,
    Run,
    WorkspaceMembership,
)
from app.db.resume_commands import ResumeCommandWriter, SqlAlchemyResumeCommandStore
from app.db.session import AsyncSessionFactory, database_session, transaction
from app.domain.errors import DomainConflictError, DomainNotFoundError, DomainValidationError
from app.domain.project_facts import (
    CandidateFactV1,
    FactEvidenceError,
    FactEvidenceV1,
    FactExtractionV1,
    MaterialRetrievalScope,
    ScopedMaterialFile,
    candidate_issues,
    check_evidence,
)
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_commands import CommandAccepted, CommandReceiptV1, ResumeCommandRequest
from app.domain.tenancy import TenantContext


class _FactCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    project_id: UUID
    import_id: UUID
    fact_id: UUID | None = None
    expected_version: int | None = None
    candidate: CandidateFactV1 | None = None
    decision: str | None = None
    attested: bool = False


async def _project(
    session: AsyncSession, tenant: TenantContext, project_id: UUID, *, edit: bool
) -> MaterialProject:
    role = await session.scalar(
        select(WorkspaceMembership.role)
        .where(
            WorkspaceMembership.workspace_id == tenant.workspace_id,
            WorkspaceMembership.user_id == tenant.actor_user_id,
            WorkspaceMembership.revoked_at.is_(None),
        )
        .with_for_update(read=True)
    )
    if role is None or (edit and role == WorkspaceRole.REVIEWER.value):
        raise DomainNotFoundError
    project = await session.scalar(
        select(MaterialProject).where(
            MaterialProject.workspace_id == tenant.workspace_id,
            MaterialProject.id == project_id,
        )
    )
    if project is None or (
        project.created_by_user_id != tenant.actor_user_id and role != WorkspaceRole.ADMIN.value
    ):
        raise DomainNotFoundError
    return project


async def _current_set(
    session: AsyncSession, tenant: TenantContext, project_id: UUID
) -> MaterialFactSet | None:
    latest = await session.scalar(
        select(MaterialImport)
        .join(
            Run,
            (Run.workspace_id == MaterialImport.workspace_id) & (Run.id == MaterialImport.run_id),
        )
        .where(
            MaterialImport.workspace_id == tenant.workspace_id,
            MaterialImport.project_id == project_id,
            MaterialImport.cache_digest.is_not(None),
            Run.status == "completed",
        )
        .order_by(MaterialImport.created_at.desc(), MaterialImport.id.desc())
        .limit(1)
    )
    if latest is None:
        return None
    return await session.scalar(
        select(MaterialFactSet)
        .where(
            MaterialFactSet.workspace_id == tenant.workspace_id,
            MaterialFactSet.project_id == project_id,
            MaterialFactSet.cache_digest == latest.cache_digest,
        )
        .order_by(MaterialFactSet.created_at.desc())
        .limit(1)
    )


async def _files_for_import(
    session: AsyncSession, tenant: TenantContext, import_id: UUID
) -> dict[UUID, MaterialSnapshotFile]:
    rows = (
        await session.scalars(
            select(MaterialSnapshotFile)
            .join(
                MaterialSnapshot,
                (MaterialSnapshot.workspace_id == MaterialSnapshotFile.workspace_id)
                & (MaterialSnapshot.id == MaterialSnapshotFile.snapshot_id),
            )
            .where(
                MaterialSnapshot.workspace_id == tenant.workspace_id,
                MaterialSnapshot.import_id == import_id,
            )
        )
    ).all()
    return {row.id: row for row in rows}


def _add_version(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    fact_id: UUID,
    version: int,
    candidate: CandidateFactV1,
    status: str,
    issues: list[str],
    actor: UUID | None,
) -> UUID:
    version_id = uuid4()
    session.add(
        MaterialFactVersion(
            id=version_id,
            workspace_id=tenant.workspace_id,
            fact_id=fact_id,
            version=version,
            claim=candidate.claim,
            kind=candidate.kind,
            conditions_json=candidate.conditions.model_dump(mode="json"),
            review_status=status,
            issues_json=[{"code": item} for item in issues],
            created_by_user_id=actor,
        )
    )
    session.add_all(
        MaterialFactEvidence(
            id=uuid4(),
            workspace_id=tenant.workspace_id,
            fact_version_id=version_id,
            snapshot_file_id=item.snapshot_file_id,
            start_line=item.start_line,
            end_line=item.end_line,
            quote=item.quote,
        )
        for item in candidate.evidence
    )
    return version_id


def _check_candidate(candidate: CandidateFactV1, files: dict[UUID, MaterialSnapshotFile]) -> None:
    allowed = {key: value.content for key, value in files.items()}
    try:
        for item in candidate.evidence:
            check_evidence(item, allowed)
    except FactEvidenceError:
        raise DomainValidationError("fact evidence is invalid") from None


async def _evidence(
    session: AsyncSession, tenant: TenantContext, version_id: UUID
) -> tuple[FactEvidenceV1, ...]:
    rows = (
        await session.scalars(
            select(MaterialFactEvidence)
            .where(
                MaterialFactEvidence.workspace_id == tenant.workspace_id,
                MaterialFactEvidence.fact_version_id == version_id,
            )
            .order_by(MaterialFactEvidence.id)
        )
    ).all()
    return tuple(
        FactEvidenceV1(
            snapshot_file_id=row.snapshot_file_id,
            start_line=row.start_line,
            end_line=row.end_line,
            quote=row.quote,
        )
        for row in rows
    )


class _FactWriter(ResumeCommandWriter):
    supported_kinds = frozenset(
        {"material_fact_add", "material_fact_revise", "material_fact_review"}
    )

    async def authorize_and_lock(
        self, session: AsyncSession, tenant: TenantContext, request: ResumeCommandRequest
    ):
        payload = request.payload
        if not isinstance(payload, _FactCommand) or request.target_id != payload.project_id:
            raise DomainValidationError("fact command is invalid")
        await _project(session, tenant, payload.project_id, edit=True)
        fact_set = await session.scalar(
            select(MaterialFactSet).where(
                MaterialFactSet.workspace_id == tenant.workspace_id,
                MaterialFactSet.project_id == payload.project_id,
                MaterialFactSet.import_id == payload.import_id,
            )
        )
        if fact_set is None:
            raise DomainNotFoundError
        if request.kind == "material_fact_add":
            if (
                payload.candidate is None
                or payload.fact_id is not None
                or payload.expected_version is not None
            ):
                raise DomainValidationError("fact addition is invalid")
            return None
        if (
            payload.fact_id is None
            or payload.expected_version is None
            or payload.expected_version < 1
        ):
            raise DomainValidationError("fact version precondition is required")
        fact = await session.scalar(
            select(MaterialFact)
            .where(
                MaterialFact.workspace_id == tenant.workspace_id,
                MaterialFact.fact_set_id == fact_set.id,
                MaterialFact.id == payload.fact_id,
            )
            .with_for_update()
        )
        if fact is None:
            raise DomainNotFoundError
        if request.kind == "material_fact_revise" and payload.candidate is None:
            raise DomainValidationError("fact correction is missing")
        if request.kind == "material_fact_review" and payload.decision not in {"confirm", "reject"}:
            raise DomainValidationError("fact review is invalid")
        return None

    async def apply(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        request: ResumeCommandRequest,
        command_id: UUID,
    ) -> CommandReceiptV1:
        payload = request.payload
        assert isinstance(payload, _FactCommand)
        fact_set = await _current_set(session, tenant, payload.project_id)
        if fact_set is None or fact_set.import_id != payload.import_id:
            raise DomainConflictError("fact set is not current")
        files = await _files_for_import(session, tenant, fact_set.import_id)
        if request.kind == "material_fact_add":
            assert payload.candidate is not None
            _check_candidate(payload.candidate, files)
            ordinal = await session.scalar(
                select(func.coalesce(func.max(MaterialFact.ordinal), -1)).where(
                    MaterialFact.workspace_id == tenant.workspace_id,
                    MaterialFact.fact_set_id == fact_set.id,
                )
            )
            fact_id = uuid4()
            session.add(
                MaterialFact(
                    id=fact_id,
                    workspace_id=tenant.workspace_id,
                    fact_set_id=fact_set.id,
                    ordinal=int(ordinal) + 1,
                    current_version=1,
                )
            )
            _add_version(
                session,
                tenant,
                fact_id=fact_id,
                version=1,
                candidate=payload.candidate,
                status="pending",
                issues=list(candidate_issues(payload.candidate)),
                actor=tenant.actor_user_id,
            )
            return CommandReceiptV1(command_id=command_id, resource_id=fact_id, status="completed")
        fact = await session.scalar(
            select(MaterialFact)
            .where(
                MaterialFact.workspace_id == tenant.workspace_id,
                MaterialFact.id == payload.fact_id,
                MaterialFact.fact_set_id == fact_set.id,
            )
            .with_for_update()
        )
        assert fact is not None
        if fact.current_version != payload.expected_version:
            raise DomainConflictError("fact version changed")
        previous = await session.scalar(
            select(MaterialFactVersion).where(
                MaterialFactVersion.workspace_id == tenant.workspace_id,
                MaterialFactVersion.fact_id == fact.id,
                MaterialFactVersion.version == fact.current_version,
            )
        )
        assert previous is not None
        if request.kind == "material_fact_revise":
            assert payload.candidate is not None
            candidate = payload.candidate
            _check_candidate(candidate, files)
            status, issues = "pending", list(candidate_issues(candidate))
        else:
            candidate = CandidateFactV1(
                claim=previous.claim,
                kind=previous.kind,
                conditions=previous.conditions_json,
                evidence=await _evidence(session, tenant, previous.id),
            )
            issues = [entry["code"] for entry in previous.issues_json]
            if payload.decision == "confirm":
                if payload.attested and candidate.kind == "personal_statement":
                    issues = [
                        item
                        for item in issues
                        if item not in {"evidence_missing", "personal_role_requires_attestation"}
                    ]
                if issues:
                    raise DomainConflictError("fact has unresolved verification questions")
                status = "confirmed"
            else:
                status = "rejected"
        fact.current_version += 1
        version_id = _add_version(
            session,
            tenant,
            fact_id=fact.id,
            version=fact.current_version,
            candidate=candidate,
            status=status,
            issues=issues,
            actor=tenant.actor_user_id,
        )
        return CommandReceiptV1(command_id=command_id, resource_id=version_id, status="completed")


class SqlAlchemyProjectFactStore:
    def __init__(self, sessions: AsyncSessionFactory) -> None:
        self.sessions = sessions
        self.commands = SqlAlchemyResumeCommandStore(sessions)

    async def existing_extraction(
        self, tenant: TenantContext, project_id: UUID, import_id: UUID, extractor_digest: str
    ) -> UUID | None:
        async with database_session(self.sessions) as session:
            await _project(session, tenant, project_id, edit=True)
            material = await session.scalar(
                select(MaterialImport).where(
                    MaterialImport.workspace_id == tenant.workspace_id,
                    MaterialImport.project_id == project_id,
                    MaterialImport.id == import_id,
                    MaterialImport.cache_digest.is_not(None),
                )
            )
            if material is None:
                raise DomainNotFoundError
            return await session.scalar(
                select(MaterialFactSet.id).where(
                    MaterialFactSet.workspace_id == tenant.workspace_id,
                    MaterialFactSet.project_id == project_id,
                    MaterialFactSet.cache_digest == material.cache_digest,
                    MaterialFactSet.extractor_digest == extractor_digest,
                )
            )

    async def retrieval_scope(
        self, tenant: TenantContext, project_imports: tuple[tuple[UUID, UUID], ...]
    ) -> MaterialRetrievalScope:
        if (
            not project_imports
            or len(project_imports) > 20
            or len(set(project_imports)) != len(project_imports)
        ):
            raise DomainValidationError("material retrieval scope is invalid")
        scoped: list[ScopedMaterialFile] = []
        async with database_session(self.sessions) as session:
            for project_id, import_id in project_imports:
                await _project(session, tenant, project_id, edit=False)
                material = await session.scalar(
                    select(MaterialImport).where(
                        MaterialImport.workspace_id == tenant.workspace_id,
                        MaterialImport.project_id == project_id,
                        MaterialImport.id == import_id,
                        MaterialImport.cache_digest.is_not(None),
                    )
                )
                if material is None:
                    raise DomainNotFoundError
                rows = (
                    await session.execute(
                        select(MaterialSnapshotFile, MaterialSnapshot.source_revision)
                        .join(
                            MaterialSnapshot,
                            (MaterialSnapshot.workspace_id == MaterialSnapshotFile.workspace_id)
                            & (MaterialSnapshot.id == MaterialSnapshotFile.snapshot_id),
                        )
                        .where(
                            MaterialSnapshot.workspace_id == tenant.workspace_id,
                            MaterialSnapshot.import_id == import_id,
                        )
                        .order_by(MaterialSnapshot.source_id, MaterialSnapshotFile.path)
                    )
                ).all()
                scoped.extend(
                    ScopedMaterialFile(
                        id=file.id,
                        path=file.path,
                        content=file.content,
                        document_id=file.document_id,
                        source_revision=revision,
                    )
                    for file, revision in rows
                )
        return MaterialRetrievalScope(files=tuple(scoped))

    async def publish_extraction(
        self,
        tenant: TenantContext,
        project_id: UUID,
        import_id: UUID,
        extractor_digest: str,
        extraction: FactExtractionV1,
        *,
        complete: bool,
    ) -> UUID:
        async with transaction(self.sessions) as session:
            await _project(session, tenant, project_id, edit=True)
            material = await session.scalar(
                select(MaterialImport).where(
                    MaterialImport.workspace_id == tenant.workspace_id,
                    MaterialImport.project_id == project_id,
                    MaterialImport.id == import_id,
                    MaterialImport.cache_digest.is_not(None),
                )
            )
            if material is None:
                raise DomainNotFoundError
            existing = await session.scalar(
                select(MaterialFactSet).where(
                    MaterialFactSet.workspace_id == tenant.workspace_id,
                    MaterialFactSet.project_id == project_id,
                    MaterialFactSet.cache_digest == material.cache_digest,
                    MaterialFactSet.extractor_digest == extractor_digest,
                )
            )
            if existing is not None:
                return existing.id
            files = await _files_for_import(session, tenant, import_id)
            allowed = {key: value.content for key, value in files.items()}
            for candidate in extraction.facts:
                for item in candidate.evidence:
                    check_evidence(item, allowed)
            set_id = uuid4()
            session.add(
                MaterialFactSet(
                    id=set_id,
                    workspace_id=tenant.workspace_id,
                    project_id=project_id,
                    import_id=import_id,
                    cache_digest=material.cache_digest,
                    extractor_digest=extractor_digest,
                    complete=complete,
                    issues_json=[{"code": value} for value in extraction.questions],
                )
            )
            await session.flush()
            for ordinal, candidate in enumerate(extraction.facts):
                fact_id = uuid4()
                session.add(
                    MaterialFact(
                        id=fact_id,
                        workspace_id=tenant.workspace_id,
                        fact_set_id=set_id,
                        ordinal=ordinal,
                        current_version=1,
                    )
                )
                _add_version(
                    session,
                    tenant,
                    fact_id=fact_id,
                    version=1,
                    candidate=candidate,
                    status="pending",
                    issues=list(candidate_issues(candidate)),
                    actor=None,
                )
            return set_id

    async def current_facts(self, tenant: TenantContext, project_id: UUID) -> dict[str, object]:
        async with database_session(self.sessions) as session:
            await _project(session, tenant, project_id, edit=False)
            fact_set = await _current_set(session, tenant, project_id)
            if fact_set is None:
                return {
                    "fact_set_id": None,
                    "import_id": None,
                    "complete": False,
                    "issues": [],
                    "facts": [],
                }
            rows = (
                await session.execute(
                    select(MaterialFact, MaterialFactVersion)
                    .join(
                        MaterialFactVersion,
                        (MaterialFactVersion.workspace_id == MaterialFact.workspace_id)
                        & (MaterialFactVersion.fact_id == MaterialFact.id)
                        & (MaterialFactVersion.version == MaterialFact.current_version),
                    )
                    .where(
                        MaterialFact.workspace_id == tenant.workspace_id,
                        MaterialFact.fact_set_id == fact_set.id,
                    )
                    .order_by(MaterialFact.ordinal)
                )
            ).all()
            facts = []
            for fact, version in rows:
                evidence = await _evidence(session, tenant, version.id)
                details = []
                for item in evidence:
                    source = await session.execute(
                        select(MaterialSnapshotFile.path, MaterialSnapshot.source_revision)
                        .join(
                            MaterialSnapshot,
                            (MaterialSnapshot.workspace_id == MaterialSnapshotFile.workspace_id)
                            & (MaterialSnapshot.id == MaterialSnapshotFile.snapshot_id),
                        )
                        .where(
                            MaterialSnapshotFile.workspace_id == tenant.workspace_id,
                            MaterialSnapshotFile.id == item.snapshot_file_id,
                        )
                    )
                    row = source.one()
                    details.append(
                        {
                            **item.model_dump(mode="json"),
                            "path": row.path,
                            "source_revision": row.source_revision,
                        }
                    )
                facts.append(
                    {
                        "id": fact.id,
                        "version_id": version.id,
                        "version": version.version,
                        "claim": version.claim,
                        "kind": version.kind,
                        "conditions": version.conditions_json,
                        "review_status": version.review_status,
                        "issues": version.issues_json,
                        "evidence": details,
                    }
                )
            return {
                "fact_set_id": fact_set.id,
                "import_id": fact_set.import_id,
                "complete": fact_set.complete,
                "issues": fact_set.issues_json,
                "facts": facts,
            }

    async def command(
        self,
        tenant: TenantContext,
        *,
        kind: str,
        project_id: UUID,
        import_id: UUID,
        request_id: UUID,
        fact_id: UUID | None = None,
        expected_version: int | None = None,
        candidate: CandidateFactV1 | None = None,
        decision: str | None = None,
        attested: bool = False,
    ) -> CommandAccepted:
        return await self.commands.accept(
            tenant=tenant,
            request=ResumeCommandRequest(
                client_request_id=request_id,
                kind=kind,
                target_id=project_id,
                payload_version=1,
                payload=_FactCommand(
                    project_id=project_id,
                    import_id=import_id,
                    fact_id=fact_id,
                    expected_version=expected_version,
                    candidate=candidate,
                    decision=decision,
                    attested=attested,
                ),
            ),
            writer=_FactWriter(),
        )

    async def search_confirmed(
        self, tenant: TenantContext, project_ids: tuple[UUID, ...], query: str
    ) -> list[dict[str, object]]:
        if (
            not project_ids
            or len(project_ids) > 20
            or len(set(project_ids)) != len(project_ids)
            or not query.strip()
            or len(query) > 200
        ):
            raise DomainValidationError("fact search scope is invalid")
        async with database_session(self.sessions) as session:
            for project_id in project_ids:
                await _project(session, tenant, project_id, edit=False)
            results = []
            for project_id in project_ids:
                fact_set = await _current_set(session, tenant, project_id)
                if fact_set is None:
                    continue
                rows = (
                    await session.execute(
                        select(MaterialFact, MaterialFactVersion)
                        .join(
                            MaterialFactVersion,
                            (MaterialFactVersion.workspace_id == MaterialFact.workspace_id)
                            & (MaterialFactVersion.fact_id == MaterialFact.id)
                            & (MaterialFactVersion.version == MaterialFact.current_version),
                        )
                        .where(
                            MaterialFact.workspace_id == tenant.workspace_id,
                            MaterialFact.fact_set_id == fact_set.id,
                            MaterialFactVersion.review_status == "confirmed",
                            MaterialFactVersion.claim.ilike(f"%{query}%"),
                        )
                        .order_by(MaterialFact.ordinal)
                        .limit(20)
                    )
                ).all()
                results.extend(
                    {
                        "project_id": project_id,
                        "fact_id": fact.id,
                        "version_id": version.id,
                        "claim": version.claim,
                        "kind": version.kind,
                    }
                    for fact, version in rows
                )
            return results[:20]


def extractor_identity(prompt_version: str, model: str) -> str:
    return hashlib.sha256(f"{prompt_version}\n{model}\n".encode()).hexdigest()
