from __future__ import annotations

import importlib
from typing import Any

import pytest
from fastapi import FastAPI

from app.auth.supabase import SupabaseActorProvider
from app.config import Settings
from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy
from app.main import app, create_app


def test_module_level_app_is_fastapi_application() -> None:
    assert isinstance(app, FastAPI)


def test_factory_returns_independent_applications() -> None:
    first_application = create_app()
    second_application = create_app()

    assert first_application is not second_application
    assert first_application.state.settings is not second_application.state.settings
    assert "/healthz" in first_application.openapi()["paths"]
    assert "/healthz" in second_application.openapi()["paths"]


def test_factory_uses_explicit_settings_without_copying() -> None:
    settings = Settings(log_level="ERROR")

    application = create_app(settings)

    assert application.state.settings is settings


def test_run_passes_configured_host_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    main_module = importlib.import_module("app.main")
    settings = Settings(api_host="0.0.0.0")
    calls: list[tuple[object, dict[str, object]]] = []

    def run_uvicorn(application: object, **kwargs: object) -> None:
        calls.append((application, kwargs))

    monkeypatch.setattr(main_module.app.state, "settings", settings)
    monkeypatch.setattr(main_module.uvicorn, "run", run_uvicorn)

    main_module.run()

    assert calls == [
        (
            main_module.app,
            {
                "host": "0.0.0.0",
                "port": 8000,
                "lifespan": "on",
                "access_log": False,
                "log_config": None,
                "workers": 1,
            },
        )
    ]


class _FakeEngine:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def dispose(self) -> None:
        self._events.append("dispose")


class _FakeProbe:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def mark_not_ready(self) -> None:
        self._events.append("not-ready")

    async def is_ready(self) -> bool:
        return True


def _install_lifespan_fakes(
    monkeypatch: pytest.MonkeyPatch, events: list[str], settings: Settings | None = None
) -> tuple[Any, Any]:
    main_module = importlib.import_module("app.main")
    engine = _FakeEngine(events)
    session_factory = object()
    probe = _FakeProbe(events)

    def create_engine(_database_url: object, *, policy: DatabasePoolPolicy) -> _FakeEngine:
        assert policy.session.component == DatabaseComponent.API
        assert policy == DatabasePoolPolicy.from_settings(
            settings or Settings(), DatabaseComponent.API
        )
        events.append("engine")
        return engine

    def create_sessions(received_engine: object) -> object:
        assert received_engine is engine
        events.append("sessions")
        return session_factory

    def create_probe(received_engine: object) -> _FakeProbe:
        assert received_engine is engine
        events.append("probe")
        return probe

    monkeypatch.setattr(main_module, "create_database_engine", create_engine)
    monkeypatch.setattr(main_module, "create_session_factory", create_sessions)
    monkeypatch.setattr(main_module, "DatabaseReadinessProbe", create_probe)
    return session_factory, probe


async def test_lifespan_creates_resources_once_and_disposes_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    session_factory, probe = _install_lifespan_fakes(monkeypatch, events)
    application = create_app()

    assert events == []
    assert not hasattr(application.state, "database_engine")

    async with application.router.lifespan_context(application):
        events.append("running")
        assert application.state.database_session_factory is session_factory
        assert application.state.readiness_probe is probe

    assert events == ["engine", "sessions", "probe", "running", "not-ready", "dispose"]
    assert application.state.database_engine is None
    assert application.state.database_session_factory is None
    assert application.state.readiness_probe is None


async def test_supabase_lifespan_composes_actor_mapping_without_eager_jwks_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_lifespan_fakes(monkeypatch, events)
    application = create_app(
        Settings(
            auth_mode="supabase",
            supabase_project_ref="abcdefghijklmnopqrst",
            supabase_publishable_key="sb_publishable_test-public-key",
            log_level="ERROR",
        )
    )

    async with application.router.lifespan_context(application):
        assert isinstance(application.state.actor_provider, SupabaseActorProvider)

    assert events == ["engine", "sessions", "probe", "not-ready", "dispose"]


async def test_lifespan_disposes_and_clears_resources_after_application_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_lifespan_fakes(monkeypatch, events)
    application = create_app()

    with pytest.raises(RuntimeError, match="injected lifespan failure"):
        async with application.router.lifespan_context(application):
            events.append("running")
            raise RuntimeError("injected lifespan failure")

    assert events == ["engine", "sessions", "probe", "running", "not-ready", "dispose"]
    assert application.state.database_engine is None
    assert application.state.database_session_factory is None
    assert application.state.readiness_probe is None


async def test_lifespan_disposes_and_clears_resources_after_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module = importlib.import_module("app.main")
    events: list[str] = []
    engine = _FakeEngine(events)

    def create_engine(_database_url: object, *, policy: DatabasePoolPolicy) -> _FakeEngine:
        assert policy.session.component == DatabaseComponent.API
        events.append("engine")
        return engine

    def fail_session_factory(received_engine: object) -> object:
        assert received_engine is engine
        events.append("sessions")
        raise RuntimeError("injected startup failure")

    monkeypatch.setattr(main_module, "create_database_engine", create_engine)
    monkeypatch.setattr(main_module, "create_session_factory", fail_session_factory)
    application = create_app()

    with pytest.raises(RuntimeError, match="injected startup failure"):
        async with application.router.lifespan_context(application):
            raise AssertionError("lifespan must not yield after startup failure")

    assert events == ["engine", "sessions", "dispose"]
    assert application.state.database_engine is None
    assert application.state.database_session_factory is None
    assert application.state.readiness_probe is None


async def test_lifespan_applies_explicit_database_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    settings = Settings(db_pool_size=3, db_statement_timeout_ms=9000)
    _install_lifespan_fakes(monkeypatch, events, settings)
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        assert application.state.database_engine is not None
    assert events[-1] == "dispose"
