from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy
from app.main import create_app

_CONFIG_ENV_NAMES = (
    "LANGFUSE_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SAMPLE_RATE",
    "LANGFUSE_SECRET_KEY",
    "DASHSCOPE_API_KEY",
    "TAVILY_API_KEY",
    "PF_API_HOST",
    "PF_AUTH_MODE",
    "PF_DATABASE_URL",
    "PF_DB_STATEMENT_TIMEOUT_MS",
    "PF_DB_LOCK_TIMEOUT_MS",
    "PF_DB_IDLE_IN_TRANSACTION_TIMEOUT_MS",
    "PF_DB_CONNECT_TIMEOUT_SECONDS",
    "PF_DB_POOL_TIMEOUT_SECONDS",
    "PF_DB_POOL_SIZE",
    "PF_DB_MAX_OVERFLOW",
    "PF_LLM_MODE",
    "PF_LOG_LEVEL",
    "PF_MATERIAL_ALIASES_FILE",
    "PF_MOCK_PORTAL_BASE_URL",
    "PF_QWEN_CHAT_MODEL",
    "PF_QWEN_EMBEDDING_DIMENSION",
    "PF_QWEN_EMBEDDING_MODEL",
    "PF_QWEN_REASONING_EFFORT",
    "PF_QWEN_WORKSPACE_ID",
    "PF_SEARCH_MODE",
    "PF_SUPABASE_JWT_AUDIENCE",
    "PF_SUPABASE_PUBLISHABLE_KEY",
    "PF_SUPABASE_PROJECT_REF",
    "PF_TRACE_MODE",
)


@pytest.fixture(autouse=True)
def _clear_config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CONFIG_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_empty_compose_material_alias_setting_keeps_import_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PF_MATERIAL_ALIASES_FILE", "")
    assert Settings().material_aliases_file is None


def test_settings_use_safe_offline_defaults() -> None:
    settings = Settings()

    assert settings.llm_mode == "fake"
    assert settings.search_mode == "fake"
    assert settings.auth_mode == "fake"
    assert settings.supabase_project_ref is None
    assert settings.supabase_publishable_key is None
    assert settings.supabase_url is None
    assert settings.supabase_jwt_audience == "authenticated"
    assert settings.supabase_issuer is None
    assert settings.supabase_jwks_url is None
    assert settings.trace_mode == "off"
    assert settings.material_aliases_file is None
    assert settings.qwen_chat_model == "qwen3.6-flash-2026-04-16"
    assert settings.qwen_reasoning_effort == "medium"
    assert settings.qwen_embedding_model == "text-embedding-v4"
    assert settings.qwen_embedding_dimension == 1536
    assert (
        settings.database_url.get_secret_value()
        == "postgresql+psycopg://pathfinder:pathfinder@127.0.0.1:5432/pathfinder"
    )
    assert settings.log_level == "INFO"
    assert settings.mock_portal_base_url == "http://127.0.0.1:8000"
    assert settings.api_host == "127.0.0.1"
    assert settings.qwen_workspace_id is None
    assert settings.qwen_api_key is None
    assert settings.tavily_api_key is None
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert settings.langfuse_base_url is None
    assert settings.langfuse_sample_rate == 1.0


_DB_FIELDS = (
    "db_statement_timeout_ms",
    "db_lock_timeout_ms",
    "db_idle_in_transaction_timeout_ms",
    "db_connect_timeout_seconds",
    "db_pool_timeout_seconds",
    "db_pool_size",
    "db_max_overflow",
)


def test_database_environment_parsing_reaches_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = ("15000", "2000", "20000", "8", "0.25", "7", "3")
    for field, value in zip(_DB_FIELDS, overrides, strict=True):
        monkeypatch.setenv(f"PF_{field.upper()}", value)
    settings = Settings()
    policy = DatabasePoolPolicy.from_settings(settings, DatabaseComponent.API)
    assert policy.session.statement_timeout_ms == 15_000
    assert policy.session.lock_timeout_ms == 2_000
    assert policy.session.idle_in_transaction_timeout_ms == 20_000
    assert policy.session.connect_timeout_seconds == 8
    assert policy.pool_timeout_seconds == 0.25
    assert policy.pool_size == 7
    assert policy.max_overflow == 3


@pytest.mark.parametrize("field", _DB_FIELDS)
@pytest.mark.parametrize("value", [True, False, -1, "invalid", "nan", "inf", float("inf")])
def test_database_settings_reject_invalid_input(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


@pytest.mark.parametrize(
    "field", [field for field in _DB_FIELDS if field != "db_pool_timeout_seconds"]
)
@pytest.mark.parametrize("value", [1.0, 1.5, "1.0", "1e1"])
def test_database_integer_settings_reject_float_coercion(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


@pytest.mark.parametrize("field", _DB_FIELDS)
def test_invalid_database_environment_fails_safely(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    canary = "E22_SYNTHETIC_SECRET_INPUT"
    monkeypatch.setenv(f"PF_{field.upper()}", canary)
    with pytest.raises(ValidationError) as captured:
        Settings()
    assert canary not in str(captured.value)
    assert canary not in repr(captured.value)


@pytest.mark.parametrize("field", _DB_FIELDS)
def test_empty_database_environment_keeps_defaults(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    expected = getattr(Settings(), field)
    monkeypatch.setenv(f"PF_{field.upper()}", "")
    assert getattr(Settings(), field) == expected


@pytest.mark.parametrize(
    "values",
    [
        dict(zip(_DB_FIELDS, (2, 1, 1, 1, 0.001, 1, 0), strict=True)),
        dict(zip(_DB_FIELDS, (300_000, 30_000, 300_000, 60, 60, 20, 20), strict=True)),
    ],
)
def test_database_settings_accept_boundaries(values: dict[str, int | float]) -> None:
    settings = Settings(**values)
    for field, expected in values.items():
        assert getattr(settings, field) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("db_statement_timeout_ms", 300_001),
        ("db_lock_timeout_ms", 30_001),
        ("db_idle_in_transaction_timeout_ms", 300_001),
        ("db_connect_timeout_seconds", 61),
        ("db_pool_timeout_seconds", 60.01),
        ("db_pool_size", 21),
        ("db_max_overflow", 21),
        *((field, 0) for field in _DB_FIELDS if field != "db_max_overflow"),
    ],
)
def test_database_settings_reject_out_of_range_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


@pytest.mark.parametrize("lock_timeout", [5_000, 5_001])
def test_database_lock_timeout_must_be_below_statement(lock_timeout: int) -> None:
    with pytest.raises(ValidationError, match="less than statement"):
        Settings(db_lock_timeout_ms=lock_timeout)


def test_invalid_database_config_prevents_app_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PF_DB_POOL_SIZE", "0")
    with pytest.raises(ValidationError):
        create_app()


def test_database_cross_field_error_does_not_expose_other_secrets() -> None:
    canary = "E22_SYNTHETIC_OTHER_SECRET"
    with pytest.raises(ValidationError) as captured:
        Settings(db_lock_timeout_ms=5_000, qwen_api_key=canary)
    assert canary not in str(captured.value)
    assert canary not in repr(captured.value)


def test_database_env_example_matches_defaults() -> None:
    example = (Path(__file__).resolve().parents[2] / ".env.example").read_text()
    entries = dict(line.split("=", 1) for line in example.splitlines() if line.startswith("PF_DB_"))
    settings = Settings(**entries)
    defaults = Settings()
    assert set(entries) == {f"PF_{field.upper()}" for field in _DB_FIELDS}
    assert all(getattr(settings, field) == getattr(defaults, field) for field in _DB_FIELDS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_mode", "invalid"),
        ("search_mode", "invalid"),
        ("auth_mode", "invalid"),
        ("trace_mode", "invalid"),
        ("log_level", "TRACE"),
        ("api_host", "api.internal"),
        ("qwen_chat_model", "different-model"),
        ("qwen_reasoning_effort", "high"),
        ("qwen_embedding_model", "different-embedding"),
        ("qwen_embedding_dimension", 1024),
    ],
)
def test_settings_reject_unknown_modes_profiles_and_log_levels(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_qwen_mode_requires_nonempty_key_and_workspace_without_leaking_them() -> None:
    with pytest.raises(
        ValidationError,
        match="DASHSCOPE_API_KEY is required when PF_LLM_MODE=qwen",
    ):
        Settings(llm_mode="qwen")

    with pytest.raises(ValidationError) as captured:
        Settings(llm_mode="qwen", qwen_api_key="   ")

    assert "DASHSCOPE_API_KEY" in str(captured.value)

    with pytest.raises(
        ValidationError,
        match="PF_QWEN_WORKSPACE_ID is required when PF_LLM_MODE=qwen",
    ):
        Settings(llm_mode="qwen", qwen_api_key="sk-test")

    canary = "QWEN-CANARY-SECRET"
    with pytest.raises(ValidationError) as captured_with_secret:
        Settings(
            llm_mode="qwen",
            qwen_api_key=canary,
            qwen_workspace_id="workspace-canary",
            log_level="TRACE",
        )

    assert canary not in str(captured_with_secret.value)


def test_qwen_credentials_are_secret_and_validate_live_mode() -> None:
    canary = "QWEN-CANARY-SECRET"

    settings = Settings(
        llm_mode="qwen",
        qwen_api_key=canary,
        qwen_workspace_id="workspace-canary",
    )

    assert settings.qwen_api_key is not None
    assert settings.qwen_api_key.get_secret_value() == canary
    assert settings.qwen_workspace_id is not None
    assert settings.qwen_workspace_id.get_secret_value() == "workspace-canary"
    assert canary not in repr(settings)
    assert canary not in str(settings.model_dump())
    assert "workspace-canary" not in repr(settings)
    assert "workspace-canary" not in str(settings.model_dump())


@pytest.mark.parametrize(
    "workspace_id",
    ["", " bad", "bad.example", "bad/path", "-bad", "bad-", "a" * 64],
)
def test_qwen_workspace_id_must_be_one_dns_label(workspace_id: str) -> None:
    with pytest.raises(ValidationError, match="PF_QWEN_WORKSPACE_ID"):
        Settings(qwen_workspace_id=workspace_id)


def test_removed_openai_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(llm_mode="openai")


def test_supabase_mode_requires_browser_config_and_derives_endpoints() -> None:
    with pytest.raises(
        ValidationError,
        match="PF_SUPABASE_PROJECT_REF is required when PF_AUTH_MODE=supabase",
    ):
        Settings(auth_mode="supabase")

    settings = Settings(
        auth_mode="supabase",
        supabase_project_ref="abcdefghijklmnopqrst",
        supabase_publishable_key="sb_publishable_test-public-key",
    )

    assert settings.supabase_jwt_audience == "authenticated"
    assert settings.supabase_issuer == ("https://abcdefghijklmnopqrst.supabase.co/auth/v1")
    assert settings.supabase_jwks_url == (
        "https://abcdefghijklmnopqrst.supabase.co/auth/v1/.well-known/jwks.json"
    )
    assert settings.supabase_url == "https://abcdefghijklmnopqrst.supabase.co"


def test_supabase_mode_requires_publishable_key() -> None:
    with pytest.raises(
        ValidationError,
        match="PF_SUPABASE_PUBLISHABLE_KEY is required when PF_AUTH_MODE=supabase",
    ):
        Settings(auth_mode="supabase", supabase_project_ref="project-ref")

    public_sample = "sb_publishable_test-key.with_symbols-1"
    settings = Settings(
        auth_mode="supabase",
        supabase_project_ref="project-ref",
        supabase_publishable_key=public_sample,
    )
    assert settings.supabase_publishable_key == public_sample


@pytest.mark.parametrize(
    "publishable_key",
    [
        "",
        " sb_publishable_key",
        "sb_publishable_key ",
        "sb_secret_private",
        "service_role",
        "eyJservice-role",
    ],
)
def test_supabase_publishable_key_rejects_blank_whitespace_and_secret_styles_without_leak(
    publishable_key: str,
) -> None:
    with pytest.raises(ValidationError) as captured:
        Settings(supabase_publishable_key=publishable_key)

    message = str(captured.value)
    if publishable_key:
        assert publishable_key not in message
    assert "secret_private" not in message
    assert "service_role" not in message


@pytest.mark.parametrize(
    "project_ref",
    ["", " bad", "bad ", "Bad", "bad.example", "bad/path", "-bad", "bad-", "a" * 64],
)
def test_supabase_project_ref_must_be_one_lowercase_dns_label(project_ref: str) -> None:
    with pytest.raises(ValidationError, match="PF_SUPABASE_PROJECT_REF"):
        Settings(supabase_project_ref=project_ref)


def test_supabase_audience_is_locked_and_settings_surfaces_contain_no_token() -> None:
    settings = Settings(
        auth_mode="supabase",
        supabase_project_ref="project-ref",
        supabase_publishable_key="sb_publishable_test-public-key",
    )

    with pytest.raises(ValidationError):
        Settings(supabase_jwt_audience="another-audience")  # type: ignore[arg-type]

    token_canary = "SUPABASE-TOKEN-CANARY"
    assert token_canary not in repr(settings)
    assert token_canary not in str(settings.model_dump())


def test_tavily_mode_requires_nonempty_key_without_leaking_it() -> None:
    with pytest.raises(
        ValidationError,
        match="TAVILY_API_KEY is required when PF_SEARCH_MODE=tavily",
    ):
        Settings(search_mode="tavily")

    with pytest.raises(ValidationError) as captured:
        Settings(search_mode="tavily", tavily_api_key="   ")

    assert "TAVILY_API_KEY is required when PF_SEARCH_MODE=tavily" in str(captured.value)

    canary = "TAVILY-CANARY-SECRET"
    with pytest.raises(ValidationError) as captured_with_secret:
        Settings(search_mode="tavily", tavily_api_key=canary, log_level="TRACE")

    assert canary not in str(captured_with_secret.value)


def test_tavily_key_is_secret_and_validates_live_mode() -> None:
    canary = "TAVILY-CANARY-SECRET"

    settings = Settings(search_mode="tavily", tavily_api_key=canary)

    assert settings.tavily_api_key is not None
    assert settings.tavily_api_key.get_secret_value() == canary
    assert canary not in repr(settings)
    assert canary not in str(settings.model_dump())


def test_langfuse_mode_requires_all_credentials_and_explicit_cloud_origin() -> None:
    with pytest.raises(ValidationError, match="LANGFUSE_PUBLIC_KEY is required"):
        Settings(trace_mode="langfuse")
    with pytest.raises(ValidationError, match="LANGFUSE_SECRET_KEY is required"):
        Settings(trace_mode="langfuse", langfuse_public_key="pk-test")
    with pytest.raises(ValidationError, match="LANGFUSE_BASE_URL is required"):
        Settings(
            trace_mode="langfuse",
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "https://cloud.langfuse.com",
        "https://us.cloud.langfuse.com/",
        "https://jp.cloud.langfuse.com:443",
        "https://hipaa.cloud.langfuse.com",
    ],
)
def test_langfuse_mode_accepts_only_locked_cloud_origins(base_url: str) -> None:
    settings = Settings(
        trace_mode="langfuse",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_base_url=base_url,
    )

    assert settings.langfuse_base_url == base_url.rstrip("/")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://cloud.langfuse.com",
        "https://example.com",
        "https://cloud.langfuse.com.example.com",
        "https://cloud.langfuse.com:444",
        "https://user@cloud.langfuse.com",
        "https://cloud.langfuse.com/api",
        "https://cloud.langfuse.com?region=us",
        "https://cloud.langfuse.com#fragment",
    ],
)
def test_langfuse_mode_rejects_non_cloud_origins(base_url: str) -> None:
    with pytest.raises(
        ValidationError,
        match="LANGFUSE_BASE_URL must be an official HTTPS Langfuse Cloud origin",
    ):
        Settings(
            trace_mode="langfuse",
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
            langfuse_base_url=base_url,
        )


@pytest.mark.parametrize("sample_rate", [-0.01, 1.01, True, float("nan")])
def test_langfuse_sample_rate_rejects_values_outside_the_closed_unit_interval(
    sample_rate: object,
) -> None:
    with pytest.raises(ValidationError):
        Settings(langfuse_sample_rate=sample_rate)


def test_langfuse_credentials_are_secret_and_validation_errors_do_not_leak() -> None:
    public_canary = "LANGFUSE-PUBLIC-CANARY"
    secret_canary = "LANGFUSE-SECRET-CANARY"
    settings = Settings(
        trace_mode="langfuse",
        langfuse_public_key=public_canary,
        langfuse_secret_key=secret_canary,
        langfuse_base_url="https://us.cloud.langfuse.com",
    )

    assert settings.langfuse_public_key is not None
    assert settings.langfuse_secret_key is not None
    assert settings.langfuse_public_key.get_secret_value() == public_canary
    assert settings.langfuse_secret_key.get_secret_value() == secret_canary
    assert public_canary not in repr(settings)
    assert secret_canary not in repr(settings)
    assert public_canary not in str(settings.model_dump())
    assert secret_canary not in str(settings.model_dump())

    with pytest.raises(ValidationError) as captured:
        Settings(
            trace_mode="langfuse",
            langfuse_public_key=public_canary,
            langfuse_secret_key=secret_canary,
            langfuse_base_url="https://example.com",
        )
    assert public_canary not in str(captured.value)
    assert secret_canary not in str(captured.value)


@pytest.mark.parametrize(
    "database_url",
    [
        "",
        "postgresql+psycopg://localhost/pathfinder",
        "postgresql+psycopg://localhost:5432",
        "postgresql://localhost:5432/pathfinder",
        "sqlite:///pathfinder.db",
    ],
)
def test_settings_reject_invalid_database_urls_without_leaking_them(
    database_url: str,
) -> None:
    with pytest.raises(ValidationError) as captured:
        Settings(database_url=database_url)

    assert "PF_DATABASE_URL must use postgresql+psycopg" in str(captured.value)
    if database_url:
        assert database_url not in str(captured.value)


def test_database_url_is_secret_and_reads_locked_environment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "DATABASE-CANARY-PASSWORD"
    database_url = f"postgresql+psycopg://pathfinder:{canary}@db.test:6432/pathfinder"
    monkeypatch.setenv("PF_DATABASE_URL", database_url)

    settings = Settings()

    assert settings.database_url.get_secret_value() == database_url
    assert canary not in repr(settings)
    assert canary not in str(settings.model_dump())


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:8000",
        "http://example.com:8000",
        "http://127.0.0.1:8000/internal",
        "http://user:pass@127.0.0.1:8000",
    ],
)
def test_mock_portal_origin_is_restricted_to_trusted_internal_http(base_url: str) -> None:
    with pytest.raises(ValidationError, match="trusted internal HTTP origin"):
        Settings(mock_portal_base_url=base_url)


def test_api_host_accepts_only_loopback_or_all_interfaces() -> None:
    assert Settings(api_host="127.0.0.1").api_host == "127.0.0.1"
    assert Settings(api_host="0.0.0.0").api_host == "0.0.0.0"

    for invalid_host in ("localhost", "api", "192.0.2.1", "::", " 0.0.0.0"):
        with pytest.raises(ValidationError):
            Settings(api_host=invalid_host)


def test_settings_read_locked_environment_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PF_API_HOST", "0.0.0.0")
    monkeypatch.setenv("PF_SEARCH_MODE", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-environment-canary")
    monkeypatch.setenv("PF_AUTH_MODE", "supabase")
    monkeypatch.setenv("PF_SUPABASE_PROJECT_REF", "environment-project")
    monkeypatch.setenv("PF_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_environment-key")
    monkeypatch.setenv("PF_SUPABASE_JWT_AUDIENCE", "authenticated")
    monkeypatch.setenv("PF_TRACE_MODE", "langfuse")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "langfuse-public-environment-canary")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "langfuse-secret-environment-canary")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.25")
    monkeypatch.setenv("PF_LOG_LEVEL", "WARNING")

    settings = Settings()

    assert settings.search_mode == "tavily"
    assert settings.auth_mode == "supabase"
    assert settings.supabase_project_ref == "environment-project"
    assert settings.supabase_publishable_key == "sb_publishable_environment-key"
    assert settings.trace_mode == "langfuse"
    assert settings.langfuse_base_url == "https://us.cloud.langfuse.com"
    assert settings.langfuse_sample_rate == 0.25
    assert settings.log_level == "WARNING"
    assert settings.api_host == "0.0.0.0"


def test_settings_do_not_load_dotenv_implicitly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text(
        "PF_LLM_MODE=qwen\nPF_AUTH_MODE=supabase\nPF_SUPABASE_PROJECT_REF=dotenv-project\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert Settings().llm_mode == "fake"
    assert Settings().auth_mode == "fake"


def test_app_factory_does_not_reread_environment_when_settings_are_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    monkeypatch.setenv("PF_LLM_MODE", "invalid")

    application = create_app(settings)

    assert application.state.settings is settings
