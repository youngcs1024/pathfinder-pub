from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import pytest

import app.cli.ingest_documents as cli_module
from app.cli.ingest_documents import (
    IngestionCommandArguments,
    authorize_and_load_documents,
    parse_command_arguments,
    run,
)
from app.domain.provisioning import WorkspaceRole
from app.domain.tenancy import TenantContext, TenantService
from app.retrieval.ingestion import (
    IngestionErrorCode,
    IngestionInputError,
    ValidatedIngestionBatch,
)


class _Resolver:
    def __init__(self, tenant: TenantContext | None) -> None:
        self.tenant = tenant

    async def resolve_tenant(self, *, workspace_id, actor_user_id):  # type: ignore[no-untyped-def]
        return self.tenant


def test_cli_parser_accepts_trusted_locators_and_explicit_files() -> None:
    workspace_id = uuid4()
    actor_user_id = uuid4()

    arguments = parse_command_arguments(
        [
            "--workspace-id",
            str(workspace_id),
            "--actor-user-id",
            str(actor_user_id),
            "resume.md",
            "preferences.txt",
        ]
    )

    assert arguments == IngestionCommandArguments(
        workspace_id=workspace_id,
        actor_user_id=actor_user_id,
        files=(Path("resume.md"), Path("preferences.txt")),
    )


async def test_authorization_precedes_document_loading() -> None:
    calls = 0

    def loader(_paths: object) -> ValidatedIngestionBatch:
        nonlocal calls
        calls += 1
        raise AssertionError("document loader must not run")

    arguments = IngestionCommandArguments(uuid4(), uuid4(), (Path("secret.txt"),))

    with pytest.raises(IngestionInputError) as raised:
        await authorize_and_load_documents(
            arguments,
            tenant_service=TenantService(_Resolver(None)),
            batch_loader=loader,
        )

    assert raised.value.code is IngestionErrorCode.WORKSPACE_ACCESS_DENIED
    assert calls == 0


async def test_authorized_membership_returns_tenant_and_loaded_batch() -> None:
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.MEMBER)
    expected = ValidatedIngestionBatch(sources=())
    arguments = IngestionCommandArguments(
        tenant.workspace_id,
        tenant.actor_user_id,
        (Path("resume.txt"),),
    )

    result = await authorize_and_load_documents(
        arguments,
        tenant_service=TenantService(_Resolver(tenant)),
        batch_loader=lambda _paths: expected,
    )

    assert result.tenant == tenant
    assert result.batch is expected


@pytest.mark.parametrize(
    ("files", "code"),
    [
        ((), IngestionErrorCode.NO_DOCUMENTS),
        (
            tuple(Path(f"{index}.txt") for index in range(11)),
            IngestionErrorCode.TOO_MANY_DOCUMENTS,
        ),
    ],
)
async def test_batch_count_is_rejected_before_authorization(files, code) -> None:  # type: ignore[no-untyped-def]
    arguments = IngestionCommandArguments(uuid4(), uuid4(), files)

    with pytest.raises(IngestionInputError) as raised:
        await authorize_and_load_documents(
            arguments,
            tenant_service=TenantService(_Resolver(None)),
        )

    assert raised.value.code is code


def test_cli_prints_stable_document_ids_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = uuid4()
    second = uuid4()

    async def ingest(_arguments):  # type: ignore[no-untyped-def]
        return first, second

    monkeypatch.setattr(cli_module, "ingest_documents_command", ingest)
    run(
        [
            "--workspace-id",
            str(uuid4()),
            "--actor-user-id",
            str(uuid4()),
            "first.md",
            "second.txt",
        ]
    )

    captured = capsys.readouterr()
    assert captured.out == f"document_id={first}\ndocument_id={second}\n"
    assert captured.err == ""


def test_cli_database_warning_is_safe_and_does_not_pollute_document_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    document_id = uuid4()
    canary = "CLI-DATABASE-QUERY-CANARY"

    async def ingest(_arguments):
        logging.getLogger("psycopg").warning("database warning: %s", canary)
        return (document_id,)

    monkeypatch.setattr(cli_module, "ingest_documents_command", ingest)
    run(["--workspace-id", str(uuid4()), "--actor-user-id", str(uuid4()), "resume.md"])

    captured = capsys.readouterr()
    assert captured.out == f"document_id={document_id}\n"
    event = json.loads(captured.err)
    assert event["event"] == "vendor.observability"
    assert event["vendor_logger"] == "psycopg"
    assert canary not in captured.out + captured.err


@pytest.mark.parametrize(
    ("input_code", "exit_code", "category"),
    [
        (IngestionErrorCode.NO_DOCUMENTS, 2, "no_documents"),
        (IngestionErrorCode.WORKSPACE_ACCESS_DENIED, 2, "workspace_access_denied"),
        (None, 1, "RuntimeError"),
    ],
)
def test_cli_prints_no_partial_success_on_command_failure(
    input_code: IngestionErrorCode | None,
    exit_code: int,
    category: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sensitive_values = (
        "synthetic document content canary",
        "/private/input-resume.md",
        "synthetic provider message",
        "synthetic-api-key",
        "postgresql://synthetic:password@database/private",
    )

    async def fail(_arguments):  # type: ignore[no-untyped-def]
        error = IngestionInputError(input_code) if input_code is not None else RuntimeError()
        error.args = (" ".join(sensitive_values),)
        raise error

    monkeypatch.setattr(cli_module, "ingest_documents_command", fail)
    with pytest.raises(SystemExit) as raised:
        run(
            [
                "--workspace-id",
                str(uuid4()),
                "--actor-user-id",
                str(uuid4()),
                sensitive_values[1],
                "second.txt",
            ]
        )

    captured = capsys.readouterr()
    assert raised.value.code == exit_code
    assert captured.out == ""
    assert captured.err == f"document ingestion failed: {category}\n"
    for value in sensitive_values:
        assert value not in captured.err


@pytest.mark.parametrize(
    "entrypoint", ["compose_authorized_ingestion_batch", "ingest_documents_command"]
)
async def test_cli_entrypoints_apply_settings_and_dispose_on_failure(
    entrypoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.config import Settings
    from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy

    settings = Settings(db_pool_size=3, db_statement_timeout_ms=9000)
    engine = SimpleNamespace(dispose=AsyncMock())

    def create_engine(url, *, policy):
        assert url == settings.database_url
        assert policy == DatabasePoolPolicy.from_settings(settings, DatabaseComponent.INGEST)
        return engine

    def fail_sessions(_engine):
        assert _engine is engine
        raise RuntimeError("stop before authorization")

    readiness = SimpleNamespace(is_ready=AsyncMock(return_value=False), mark_not_ready=lambda: None)
    monkeypatch.setattr(cli_module, "create_database_engine", create_engine)
    monkeypatch.setattr(cli_module, "create_session_factory", fail_sessions)
    monkeypatch.setattr(cli_module, "DatabaseReadinessProbe", lambda _engine: readiness)
    arguments = IngestionCommandArguments(uuid4(), uuid4(), (Path("unused.txt"),))
    expected = (
        "stop before authorization"
        if entrypoint == "compose_authorized_ingestion_batch"
        else "database is unavailable"
    )
    with pytest.raises(RuntimeError, match=expected):
        await getattr(cli_module, entrypoint)(arguments, settings=settings)
    engine.dispose.assert_awaited_once()
