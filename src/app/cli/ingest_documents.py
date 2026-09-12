from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from app.config import Settings
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.readiness import DatabaseReadinessProbe
from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.errors import DomainNotFoundError
from app.domain.tenancy import TenantContext, TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeChatModel, FakeEmbeddingModel
from app.llm.invocations import LLMInvocationContext
from app.llm.qwen_adapters import create_qwen_adapters
from app.obs.langfuse import build_trace_sink
from app.obs.logging import configure_logging
from app.retrieval.chunking import normalize_and_chunk_batch
from app.retrieval.documents import DocumentIngestionService
from app.retrieval.ingestion import (
    MAX_DOCUMENTS_PER_COMMAND,
    IngestionErrorCode,
    IngestionInputError,
    ValidatedIngestionBatch,
    validate_and_load_documents,
)

BatchLoader = Callable[[Sequence[Path]], ValidatedIngestionBatch]


@dataclass(frozen=True, slots=True)
class IngestionCommandArguments:
    workspace_id: UUID
    actor_user_id: UUID
    files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class AuthorizedIngestionBatch:
    tenant: TenantContext
    batch: ValidatedIngestionBatch


def _uuid_argument(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a UUID") from None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pathfinder-ingest-documents",
        description="Validate explicit local Markdown/text inputs for a workspace member.",
    )
    parser.add_argument("--workspace-id", required=True, type=_uuid_argument)
    parser.add_argument("--actor-user-id", required=True, type=_uuid_argument)
    parser.add_argument("files", nargs="*", type=Path)
    return parser


def parse_command_arguments(argv: Sequence[str] | None = None) -> IngestionCommandArguments:
    namespace = build_argument_parser().parse_args(argv)
    return IngestionCommandArguments(
        workspace_id=namespace.workspace_id,
        actor_user_id=namespace.actor_user_id,
        files=tuple(namespace.files),
    )


async def authorize_and_load_documents(
    arguments: IngestionCommandArguments,
    *,
    tenant_service: TenantService,
    batch_loader: BatchLoader = validate_and_load_documents,
) -> AuthorizedIngestionBatch:
    if not arguments.files:
        raise IngestionInputError(IngestionErrorCode.NO_DOCUMENTS)
    if len(arguments.files) > MAX_DOCUMENTS_PER_COMMAND:
        raise IngestionInputError(IngestionErrorCode.TOO_MANY_DOCUMENTS)

    try:
        tenant = await tenant_service.resolve_tenant(
            workspace_id=arguments.workspace_id,
            actor_user_id=arguments.actor_user_id,
        )
    except DomainNotFoundError:
        raise IngestionInputError(IngestionErrorCode.WORKSPACE_ACCESS_DENIED) from None

    return AuthorizedIngestionBatch(
        tenant=tenant,
        batch=batch_loader(arguments.files),
    )


async def compose_authorized_ingestion_batch(
    arguments: IngestionCommandArguments,
    *,
    settings: Settings | None = None,
    batch_loader: BatchLoader = validate_and_load_documents,
) -> AuthorizedIngestionBatch:
    """Create and clean up the Step 5.1 CLI dependencies around one validation run."""

    runtime_settings = settings or Settings()
    engine = create_database_engine(
        runtime_settings.database_url,
        policy=DatabasePoolPolicy.from_settings(runtime_settings, DatabaseComponent.INGEST),
    )
    try:
        tenant_service = TenantService(SqlAlchemyTenantResolver(create_session_factory(engine)))
        return await authorize_and_load_documents(
            arguments,
            tenant_service=tenant_service,
            batch_loader=batch_loader,
        )
    finally:
        await engine.dispose()


async def ingest_documents_command(
    arguments: IngestionCommandArguments,
    *,
    settings: Settings | None = None,
) -> tuple[UUID, ...]:
    """Run the authorized Step 5.3 pipeline and return IDs without printing partial success."""

    runtime_settings = settings or Settings()
    engine = create_database_engine(
        runtime_settings.database_url,
        policy=DatabasePoolPolicy.from_settings(runtime_settings, DatabaseComponent.INGEST),
    )
    readiness = DatabaseReadinessProbe(engine)
    qwen_bundle: Any | None = None
    trace_sink: Any | None = None
    try:
        if not await readiness.is_ready():
            raise RuntimeError("database is unavailable or not at the required revision")
        session_factory = create_session_factory(engine)
        authorized = await authorize_and_load_documents(
            arguments,
            tenant_service=TenantService(SqlAlchemyTenantResolver(session_factory)),
        )
        prepared = normalize_and_chunk_batch(authorized.batch)
        trace_sink = build_trace_sink(runtime_settings)
        if runtime_settings.llm_mode == "qwen":
            assert runtime_settings.qwen_api_key is not None
            assert runtime_settings.qwen_workspace_id is not None
            qwen_bundle = create_qwen_adapters(
                api_key=runtime_settings.qwen_api_key,
                workspace_id=runtime_settings.qwen_workspace_id,
            )
            chat_adapter = qwen_bundle.chat
            embedding_adapter = qwen_bundle.embedding
            provider = "qwen"
        else:
            chat_adapter = FakeChatModel()
            embedding_adapter = FakeEmbeddingModel()
            provider = "fake"
        factory = LLMFactory(
            recorder=SqlAlchemyInvocationRecorder(session_factory),
            chat_adapter=chat_adapter,
            embedding_adapter=embedding_adapter,
            trace_sink=trace_sink,
            provider=provider,
        )
        service = DocumentIngestionService(
            repository=SqlAlchemyDocumentRepository(session_factory),
            embedding=factory.create_embedding_model(
                LLMInvocationContext(
                    workspace_id=authorized.tenant.workspace_id,
                    actor_user_id=authorized.tenant.actor_user_id,
                    request_id=None,
                    run_id=None,
                )
            ),
        )
        return await service.ingest(
            tenant=authorized.tenant,
            batch=prepared,
        )
    finally:
        readiness.mark_not_ready()
        try:
            if qwen_bundle is not None:
                await qwen_bundle.aclose()
        finally:
            try:
                if trace_sink is not None:
                    shutdown = getattr(trace_sink, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
            finally:
                await engine.dispose()


def run(argv: Sequence[str] | None = None) -> None:
    try:
        configure_logging(log_level="WARNING", stream=sys.stderr)
        document_ids = asyncio.run(ingest_documents_command(parse_command_arguments(argv)))
    except IngestionInputError as error:
        print(
            f"document ingestion failed: {error.code.value}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    except Exception as error:
        print(
            f"document ingestion failed: {type(error).__name__}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    for document_id in document_ids:
        print(f"document_id={document_id}")


if __name__ == "__main__":
    run()
