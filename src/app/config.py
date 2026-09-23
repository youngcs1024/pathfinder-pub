import re
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.db.runtime_policy import (
    ConnectTimeoutSeconds,
    IdleInTransactionTimeoutMs,
    LockTimeoutMs,
    MaxOverflow,
    PoolSize,
    PoolTimeoutSeconds,
    StatementTimeoutMs,
    validate_timeout_order,
)

LLMMode = Literal["fake", "qwen"]
SearchMode = Literal["fake", "tavily"]
AuthMode = Literal["fake", "supabase"]
TraceMode = Literal["off", "langfuse"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_LANGFUSE_CLOUD_HOSTS = frozenset(
    {
        "cloud.langfuse.com",
        "hipaa.cloud.langfuse.com",
        "jp.cloud.langfuse.com",
        "us.cloud.langfuse.com",
    }
)
_QWEN_WORKSPACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_SUPABASE_PROJECT_REF_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_SUPABASE_PUBLISHABLE_KEY_PATTERN = re.compile(r"^sb_publishable_[A-Za-z0-9._-]+$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=None,
        env_ignore_empty=True,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        populate_by_name=True,
        validate_default=True,
    )

    llm_mode: LLMMode = Field(default="fake", validation_alias="PF_LLM_MODE")
    search_mode: SearchMode = Field(default="fake", validation_alias="PF_SEARCH_MODE")
    auth_mode: AuthMode = Field(default="fake", validation_alias="PF_AUTH_MODE")
    trace_mode: TraceMode = Field(default="off", validation_alias="PF_TRACE_MODE")
    supabase_project_ref: str | None = Field(
        default=None,
        validation_alias="PF_SUPABASE_PROJECT_REF",
    )
    supabase_publishable_key: str | None = Field(
        default=None,
        validation_alias="PF_SUPABASE_PUBLISHABLE_KEY",
        repr=False,
    )
    supabase_jwt_audience: Literal["authenticated"] = Field(
        default="authenticated",
        validation_alias="PF_SUPABASE_JWT_AUDIENCE",
    )

    qwen_chat_model: Literal["qwen3.6-flash-2026-04-16"] = Field(
        default="qwen3.6-flash-2026-04-16",
        validation_alias="PF_QWEN_CHAT_MODEL",
    )
    qwen_reasoning_effort: Literal["medium"] = Field(
        default="medium",
        validation_alias="PF_QWEN_REASONING_EFFORT",
    )
    qwen_embedding_model: Literal["text-embedding-v4"] = Field(
        default="text-embedding-v4",
        validation_alias="PF_QWEN_EMBEDDING_MODEL",
    )
    qwen_embedding_dimension: Literal[1536] = Field(
        default=1536,
        validation_alias="PF_QWEN_EMBEDDING_DIMENSION",
    )
    database_url: SecretStr = Field(
        default="postgresql+psycopg://pathfinder:pathfinder@127.0.0.1:5432/pathfinder",
        validation_alias="PF_DATABASE_URL",
        repr=False,
    )
    db_statement_timeout_ms: StatementTimeoutMs = Field(
        default=5_000, validation_alias="PF_DB_STATEMENT_TIMEOUT_MS"
    )
    db_lock_timeout_ms: LockTimeoutMs = Field(
        default=1_000, validation_alias="PF_DB_LOCK_TIMEOUT_MS"
    )
    db_idle_in_transaction_timeout_ms: IdleInTransactionTimeoutMs = Field(
        default=10_000, validation_alias="PF_DB_IDLE_IN_TRANSACTION_TIMEOUT_MS"
    )
    db_connect_timeout_seconds: ConnectTimeoutSeconds = Field(
        default=5, validation_alias="PF_DB_CONNECT_TIMEOUT_SECONDS"
    )
    db_pool_timeout_seconds: PoolTimeoutSeconds = Field(
        default=2.0, validation_alias="PF_DB_POOL_TIMEOUT_SECONDS"
    )
    db_pool_size: PoolSize = Field(default=5, validation_alias="PF_DB_POOL_SIZE")
    db_max_overflow: MaxOverflow = Field(default=0, validation_alias="PF_DB_MAX_OVERFLOW")
    mock_portal_base_url: str = Field(
        default="http://127.0.0.1:8000",
        validation_alias="PF_MOCK_PORTAL_BASE_URL",
    )
    api_host: Literal["127.0.0.1", "0.0.0.0"] = Field(
        default="127.0.0.1",
        validation_alias="PF_API_HOST",
    )
    log_level: LogLevel = Field(default="INFO", validation_alias="PF_LOG_LEVEL")
    material_aliases_file: Path | None = Field(
        default=None, validation_alias="PF_MATERIAL_ALIASES_FILE"
    )
    qwen_workspace_id: SecretStr | None = Field(
        default=None,
        validation_alias="PF_QWEN_WORKSPACE_ID",
        repr=False,
    )
    qwen_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="DASHSCOPE_API_KEY",
        repr=False,
    )
    tavily_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="TAVILY_API_KEY",
        repr=False,
    )
    langfuse_public_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGFUSE_PUBLIC_KEY",
        repr=False,
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGFUSE_SECRET_KEY",
        repr=False,
    )
    langfuse_base_url: str | None = Field(
        default=None,
        validation_alias="LANGFUSE_BASE_URL",
    )
    langfuse_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        validation_alias="LANGFUSE_SAMPLE_RATE",
    )

    @model_validator(mode="after")
    def validate_database_timeouts(self) -> Self:
        validate_timeout_order(self.db_statement_timeout_ms, self.db_lock_timeout_ms)
        return self

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        raw_value = value.get_secret_value().strip()
        try:
            parsed = urlsplit(raw_value)
            has_valid_location = parsed.hostname is not None and parsed.port is not None
        except ValueError:
            has_valid_location = False
            parsed = None

        if (
            parsed is None
            or parsed.scheme != "postgresql+psycopg"
            or not has_valid_location
            or not parsed.path.strip("/")
        ):
            raise ValueError(
                "PF_DATABASE_URL must use postgresql+psycopg with host, port, and database"
            )
        return SecretStr(raw_value)

    @field_validator("qwen_workspace_id")
    @classmethod
    def validate_qwen_workspace_id(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        untrimmed_value = value.get_secret_value()
        raw_value = untrimmed_value.strip()
        if untrimmed_value != raw_value or _QWEN_WORKSPACE_ID_PATTERN.fullmatch(raw_value) is None:
            raise ValueError("PF_QWEN_WORKSPACE_ID must be one valid DNS label")
        return SecretStr(raw_value)

    @field_validator("supabase_project_ref")
    @classmethod
    def validate_supabase_project_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        raw_value = value.strip()
        if value != raw_value or _SUPABASE_PROJECT_REF_PATTERN.fullmatch(raw_value) is None:
            raise ValueError("PF_SUPABASE_PROJECT_REF must be one lowercase DNS label")
        return raw_value

    @field_validator("supabase_publishable_key")
    @classmethod
    def validate_supabase_publishable_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value != value.strip() or _SUPABASE_PUBLISHABLE_KEY_PATTERN.fullmatch(value) is None:
            raise ValueError(
                "PF_SUPABASE_PUBLISHABLE_KEY must be a current browser publishable key"
            )
        return value

    @field_validator("mock_portal_base_url")
    @classmethod
    def validate_mock_portal_base_url(cls, value: str) -> str:
        raw_value = value.strip().rstrip("/")
        try:
            parsed = urlsplit(raw_value)
            port = parsed.port
        except ValueError:
            parsed = None
            port = None
        if (
            parsed is None
            or parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "api"}
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("PF_MOCK_PORTAL_BASE_URL must be a trusted internal HTTP origin")
        return raw_value

    @field_validator("qwen_api_key")
    @classmethod
    def validate_qwen_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        raw_value = value.get_secret_value()
        if not raw_value or raw_value != raw_value.strip():
            raise ValueError("DASHSCOPE_API_KEY must be nonempty without surrounding whitespace")
        return SecretStr(raw_value)

    @field_validator("langfuse_base_url")
    @classmethod
    def validate_langfuse_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        raw_value = value.strip().rstrip("/")
        try:
            parsed = urlsplit(raw_value)
            port = parsed.port
        except ValueError:
            parsed = None
            port = None
        if (
            parsed is None
            or parsed.scheme != "https"
            or parsed.hostname not in _LANGFUSE_CLOUD_HOSTS
            or port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("LANGFUSE_BASE_URL must be an official HTTPS Langfuse Cloud origin")
        return raw_value

    @field_validator("langfuse_sample_rate", mode="before")
    @classmethod
    def validate_langfuse_sample_rate(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("LANGFUSE_SAMPLE_RATE must be a number between 0 and 1")
        return value

    @model_validator(mode="after")
    def require_live_provider_keys(self) -> Self:
        if self.auth_mode == "supabase" and self.supabase_project_ref is None:
            raise ValueError("PF_SUPABASE_PROJECT_REF is required when PF_AUTH_MODE=supabase")
        if self.auth_mode == "supabase" and self.supabase_publishable_key is None:
            raise ValueError("PF_SUPABASE_PUBLISHABLE_KEY is required when PF_AUTH_MODE=supabase")
        if self.llm_mode == "qwen":
            qwen_secret = self.qwen_api_key
            if qwen_secret is None or not qwen_secret.get_secret_value().strip():
                raise ValueError("DASHSCOPE_API_KEY is required when PF_LLM_MODE=qwen")
            workspace_id = self.qwen_workspace_id
            if workspace_id is None or not workspace_id.get_secret_value().strip():
                raise ValueError("PF_QWEN_WORKSPACE_ID is required when PF_LLM_MODE=qwen")
        tavily_secret = self.tavily_api_key
        if self.search_mode == "tavily" and (
            tavily_secret is None or not tavily_secret.get_secret_value().strip()
        ):
            raise ValueError("TAVILY_API_KEY is required when PF_SEARCH_MODE=tavily")
        if self.trace_mode == "langfuse":
            public_key = self.langfuse_public_key
            secret_key = self.langfuse_secret_key
            if public_key is None or not public_key.get_secret_value().strip():
                raise ValueError("LANGFUSE_PUBLIC_KEY is required when PF_TRACE_MODE=langfuse")
            if secret_key is None or not secret_key.get_secret_value().strip():
                raise ValueError("LANGFUSE_SECRET_KEY is required when PF_TRACE_MODE=langfuse")
            if self.langfuse_base_url is None:
                raise ValueError("LANGFUSE_BASE_URL is required when PF_TRACE_MODE=langfuse")
        return self

    @property
    def supabase_issuer(self) -> str | None:
        if self.supabase_project_ref is None:
            return None
        return f"https://{self.supabase_project_ref}.supabase.co/auth/v1"

    @property
    def supabase_url(self) -> str | None:
        if self.supabase_project_ref is None:
            return None
        return f"https://{self.supabase_project_ref}.supabase.co"

    @property
    def supabase_jwks_url(self) -> str | None:
        issuer = self.supabase_issuer
        if issuer is None:
            return None
        return f"{issuer}/.well-known/jwks.json"
