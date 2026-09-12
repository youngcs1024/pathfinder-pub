from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.db.runtime_policy import (
    DatabaseComponent,
    DatabasePoolPolicy,
    DatabaseSessionPolicy,
    checkpoint_setup_policy,
)


@pytest.fixture(autouse=True)
def _offline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in Settings.model_fields.values():
        if isinstance(field.validation_alias, str):
            monkeypatch.delenv(field.validation_alias, raising=False)


def test_defaults_and_component_identity() -> None:
    settings = Settings()
    for component in DatabaseComponent:
        session = DatabaseSessionPolicy.from_settings(settings, component)
        assert session.application_name == component.value
        assert session.statement_timeout_ms == 5_000
        assert session.lock_timeout_ms == 1_000
        assert session.idle_in_transaction_timeout_ms == 10_000
        assert session.connect_timeout_seconds == 5
        if component != DatabaseComponent.CHECKPOINT:
            pool = DatabasePoolPolicy.from_settings(settings, component)
            assert pool.session == session
            assert pool.pool_size == 5
            assert pool.max_overflow == 0
            assert pool.pool_timeout_seconds == 2.0
            assert pool.pool_pre_ping is True
    assert {component.value for component in DatabaseComponent} == {
        "pathfinder-api",
        "pathfinder-worker",
        "pathfinder-ingest",
        "pathfinder-checkpoint",
        "pathfinder-worker-healthcheck",
    }


def test_setup_is_independent_of_runtime_and_pool_overrides() -> None:
    settings = Settings(
        db_statement_timeout_ms=300_000,
        db_lock_timeout_ms=30_000,
        db_idle_in_transaction_timeout_ms=123_000,
        db_connect_timeout_seconds=12,
        db_pool_size=20,
        db_max_overflow=20,
    )
    setup = checkpoint_setup_policy(settings)
    runtime = DatabaseSessionPolicy.from_settings(settings, DatabaseComponent.CHECKPOINT)
    assert checkpoint_setup_policy(runtime) == setup
    assert setup.component == DatabaseComponent.CHECKPOINT
    assert setup.statement_timeout_ms == 30_000
    assert setup.lock_timeout_ms == 5_000
    assert setup.idle_in_transaction_timeout_ms == 123_000
    assert setup.connect_timeout_seconds == 12
    assert runtime.statement_timeout_ms == 300_000
    assert runtime.lock_timeout_ms == 30_000
    assert not {"pool_size", "max_overflow", "pool_timeout_seconds"} & setup.model_dump().keys()
    with pytest.raises(ValueError, match="independent session"):
        DatabasePoolPolicy.from_settings(settings, DatabaseComponent.CHECKPOINT)
    with pytest.raises(ValidationError, match="independent session"):
        DatabasePoolPolicy(session=runtime)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("statement_timeout_ms", 0),
        ("statement_timeout_ms", 300_001),
        ("lock_timeout_ms", 30_001),
        ("lock_timeout_ms", 5_000),
        ("idle_in_transaction_timeout_ms", 300_001),
        ("connect_timeout_seconds", 61),
        ("connect_timeout_seconds", True),
        ("connect_timeout_seconds", 5.0),
        ("connect_timeout_seconds", "5.0"),
        ("component", "untrusted-component"),
        ("application_name", "untrusted-component"),
        ("pool_size", 5),
    ],
)
def test_direct_session_construction_rejects_invalid_values(field: str, value: object) -> None:
    values = {"component": DatabaseComponent.API, field: value}
    with pytest.raises(ValidationError):
        DatabaseSessionPolicy.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pool_size", 0),
        ("pool_size", 21),
        ("max_overflow", -1),
        ("max_overflow", 21),
        ("max_overflow", False),
        ("pool_timeout_seconds", 0),
        ("pool_timeout_seconds", 60.01),
        ("pool_timeout_seconds", float("inf")),
        ("pool_timeout_seconds", float("nan")),
        ("pool_timeout_seconds", True),
        ("pool_pre_ping", False),
    ],
)
def test_direct_pool_construction_rejects_invalid_values(field: str, value: object) -> None:
    values = {"session": DatabaseSessionPolicy(component=DatabaseComponent.API), field: value}
    with pytest.raises(ValidationError):
        DatabasePoolPolicy.model_validate(values)


def test_policy_is_deeply_immutable_and_secret_free() -> None:
    canary = "E22_SYNTHETIC_PRIVATE_VALUE"
    settings = Settings(
        database_url=f"postgresql+psycopg://user:{canary}@127.0.0.1:5432/test",
        qwen_api_key=canary,
    )
    policy = DatabasePoolPolicy.from_settings(settings, DatabaseComponent.API)
    assert canary not in repr(policy)
    assert canary not in policy.model_dump_json()
    for target, field, value in (
        (policy, "pool_size", 10),
        (policy.session, "statement_timeout_ms", 10_000),
        (policy.session, "component", DatabaseComponent.WORKER),
    ):
        with pytest.raises(ValidationError, match="frozen"):
            setattr(target, field, value)
    with pytest.raises(ValidationError) as captured:
        DatabaseSessionPolicy.model_validate({"component": canary})
    assert canary not in str(captured.value)
    assert canary not in repr(captured.value)
