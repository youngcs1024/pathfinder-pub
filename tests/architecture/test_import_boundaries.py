from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_SOURCE_ROOT = PROJECT_ROOT / "src" / "app"

AGENTS_ALLOWED_INTERNAL_IMPORTS = (
    "app.agents",
    "app.domain",
    "app.llm",
    "app.tools",
)
AGENTS_FORBIDDEN_EXTERNAL_IMPORTS = (
    "aiohttp",
    "fastapi",
    "http.client",
    "httpx",
    "requests",
    "socket",
    "starlette",
    "urllib.request",
    "urllib3",
)
AGENT_FRAMEWORK_IMPORTS = (
    "langchain",
    "langchain_core",
    "langgraph",
    "langsmith",
)
CREATE_AGENT_FRAMEWORK_ADAPTER_MODULE = ("app", "agents", "create_agent_loop")
RESEARCH_GRAPH_FRAMEWORK_ADAPTER_MODULE = ("app", "agents", "research_graph")
POSTGRES_CHECKPOINT_ADAPTER_MODULE = ("app", "db", "checkpoints")
QWEN_LLM_ADAPTER_MODULE = ("app", "llm", "qwen_adapters")
AGENT_FRAMEWORK_ALLOWED_MODULES = {
    "langchain": frozenset({CREATE_AGENT_FRAMEWORK_ADAPTER_MODULE}),
    "langchain_core": frozenset(
        {
            CREATE_AGENT_FRAMEWORK_ADAPTER_MODULE,
            QWEN_LLM_ADAPTER_MODULE,
        }
    ),
    "langgraph": frozenset(
        {
            CREATE_AGENT_FRAMEWORK_ADAPTER_MODULE,
            POSTGRES_CHECKPOINT_ADAPTER_MODULE,
            RESEARCH_GRAPH_FRAMEWORK_ADAPTER_MODULE,
        }
    ),
    "langsmith": frozenset({CREATE_AGENT_FRAMEWORK_ADAPTER_MODULE}),
}
API_FORBIDDEN_INTERNAL_IMPORTS = (
    "app.db",
    "app.worker",
    "app.agents",
    "app.llm",
    "app.tools",
    "app.retrieval",
)
API_FORBIDDEN_EXTERNAL_IMPORTS = (
    "alembic",
    "langchain_openai",
    "langgraph",
    "openai",
    "psycopg",
    "sqlalchemy",
    "tavily",
    "tavily_python",
)
DOMAIN_FORBIDDEN_INTERNAL_IMPORTS = (
    "app.agents",
    "app.api",
    "app.db",
    "app.main",
    "app.tools",
    "app.worker",
)
DOMAIN_FORBIDDEN_EXTERNAL_IMPORTS = (
    "alembic",
    "fastapi",
    "langgraph",
    "psycopg",
    "sqlalchemy",
    "starlette",
)
DB_FORBIDDEN_INTERNAL_IMPORTS = (
    "app.agents",
    "app.api",
    "app.llm",
    "app.main",
    "app.retrieval",
    "app.tools",
    "app.worker",
)
DB_FORBIDDEN_EXTERNAL_IMPORTS = (
    "fastapi",
    "starlette",
    "testcontainers",
)
LLM_FORBIDDEN_INTERNAL_IMPORTS = (
    "app.agents",
    "app.api",
    "app.db",
    "app.main",
    "app.retrieval",
    "app.tools",
    "app.worker",
)
RETRIEVAL_FORBIDDEN_INTERNAL_IMPORTS = (
    "app.obs",
    "app.agents",
    "app.api",
    "app.db",
    "app.main",
    "app.tools",
    "app.worker",
)
RETRIEVAL_FORBIDDEN_EXTERNAL_IMPORTS = (
    "alembic",
    "fastapi",
    "langchain_openai",
    "openai",
    "psycopg",
    "sqlalchemy",
    "starlette",
)
TOOLS_FORBIDDEN_INTERNAL_IMPORTS = (
    "app.obs",
    "app.agents",
    "app.api",
    "app.db",
    "app.main",
    "app.worker",
)
WORKER_FORBIDDEN_EXTERNAL_IMPORTS = (
    "fastapi",
    "langchain",
    "langchain_core",
    "langgraph",
    "starlette",
)
MOCK_PORTAL_FORBIDDEN_INTERNAL_IMPORTS = (
    "app.agents",
    "app.db",
    "app.worker",
)
DATABASE_LIBRARY_IMPORTS = ("alembic", "pgvector", "psycopg", "sqlalchemy")
OPENAI_IMPORTS = ("langchain_openai", "openai")
TAVILY_IMPORTS = ("tavily", "tavily_python")
JWT_IMPORTS = ("jwt",)
LANGFUSE_IMPORTS = ("langfuse",)
FAKE_FORBIDDEN_IMPORTS = (
    "aiohttp",
    "dotenv",
    "httpx",
    "langchain_openai",
    "langfuse",
    "openai",
    "os",
    "requests",
    "socket",
    "supabase",
    "tavily",
    "tavily_python",
    "urllib.request",
    "urllib3",
)
STRUCTLOG_IMPORTS = ("structlog",)
PYDANTIC_SETTINGS_IMPORTS = ("pydantic_settings",)
UVICORN_IMPORTS = ("uvicorn",)


@dataclass(frozen=True, order=True)
class Violation:
    relative_path: str
    lineno: int
    rule: str
    detail: str


def _is_import_prefix(target: str, prefix: str) -> bool:
    return target == prefix or target.startswith(f"{prefix}.")


def _is_module_within(module: tuple[str, ...], parent: tuple[str, ...]) -> bool:
    return module[: len(parent)] == parent


def _module_for_path(path: Path, source_root: Path) -> tuple[tuple[str, ...], bool]:
    relative_parts = path.relative_to(source_root).with_suffix("").parts
    module = ("app", *relative_parts)
    is_package = module[-1] == "__init__"
    if is_package:
        module = module[:-1]
    return module, is_package


def _from_import_base(
    module: tuple[str, ...], is_package: bool, node: ast.ImportFrom
) -> tuple[str, ...]:
    imported_module = tuple(node.module.split(".")) if node.module else ()
    if node.level == 0:
        return imported_module

    package = module if is_package else module[:-1]
    retained_parts = len(package) - node.level + 1
    anchor = package[: max(retained_parts, 0)]
    return (*anchor, *imported_module)


def _import_targets(
    module: tuple[str, ...], is_package: bool, node: ast.Import | ast.ImportFrom
) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)

    base = _from_import_base(module, is_package, node)
    targets: list[str] = []
    if base:
        targets.append(".".join(base))
    targets.extend(
        ".".join((*base, *alias.name.split("."))) for alias in node.names if alias.name != "*"
    )
    return tuple(dict.fromkeys(targets))


def _fastapi_bindings(tree: ast.AST) -> tuple[set[str], set[str]]:
    constructors: set[str] = set()
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "fastapi":
                    modules.add(alias.asname or "fastapi")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "fastapi":
            for alias in node.names:
                if alias.name == "FastAPI":
                    constructors.add(alias.asname or alias.name)

    return constructors, modules


def _is_fastapi_constructor(call: ast.Call, constructors: set[str], modules: set[str]) -> bool:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id in constructors
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "FastAPI"
        and isinstance(function.value, ast.Name)
        and function.value.id in modules
    )


def find_violations(source_root: Path) -> list[Violation]:
    if not source_root.is_dir():
        raise ValueError(f"application source root is not a directory: {source_root}")

    violations: list[Violation] = []
    seen: set[tuple[str, int, str]] = set()

    for path in sorted(source_root.rglob("*.py")):
        relative_path = path.relative_to(source_root).as_posix()
        module, is_package = _module_for_path(path, source_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        def add_violation(
            node: ast.AST,
            rule: str,
            detail: str,
            current_relative_path: str = relative_path,
        ) -> None:
            key = (current_relative_path, node.lineno, rule)
            if key not in seen:
                seen.add(key)
                violations.append(
                    Violation(
                        relative_path=current_relative_path,
                        lineno=node.lineno,
                        rule=rule,
                        detail=detail,
                    )
                )

        is_agents = _is_module_within(module, ("app", "agents"))
        is_auth = _is_module_within(module, ("app", "auth"))
        is_api = _is_module_within(module, ("app", "api"))
        is_domain = _is_module_within(module, ("app", "domain"))
        is_db = _is_module_within(module, ("app", "db"))
        is_llm = _is_module_within(module, ("app", "llm"))
        is_mock_portal = _is_module_within(module, ("app", "mock_portal"))
        is_retrieval = _is_module_within(module, ("app", "retrieval"))
        is_tools = _is_module_within(module, ("app", "tools"))
        is_worker = _is_module_within(module, ("app", "worker"))
        is_fake_provider = (is_llm or is_tools) and any(
            part == "fake" or part.startswith("fake_") for part in module[2:]
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                add_violation(node, "bare-except", "bare except is forbidden")

            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "handler"
                and module != ("app", "tools", "registry")
            ):
                add_violation(
                    node,
                    "tool-handler-called-outside-registry",
                    ".".join(module),
                )

            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue

            for target in _import_targets(module, is_package, node):
                if _is_import_prefix(target, "app.tools.mcp_experiment") and not _is_module_within(
                    module, ("app", "tools", "mcp_experiment")
                ):
                    add_violation(node, "experiment-import-from-production", target)

                if (
                    _is_import_prefix(target, "mcp") or _is_import_prefix(target, "mcp_types")
                ) and not _is_module_within(module, ("app", "tools", "mcp_experiment")):
                    add_violation(node, "mcp-import-outside-experiment", target)

                if any(
                    _is_import_prefix(target, prefix) for prefix in LANGFUSE_IMPORTS
                ) and not _is_module_within(module, ("app", "obs")):
                    add_violation(node, "langfuse-import-outside-obs", target)

                if (
                    module == ("app", "domain", "tracing")
                    and target.split(".", 1)[0] not in sys.stdlib_module_names
                ):
                    add_violation(node, "domain-tracing-non-stdlib-import", target)

                if (
                    is_agents
                    and target != "app"
                    and _is_import_prefix(target, "app")
                    and not any(
                        _is_import_prefix(target, prefix)
                        for prefix in AGENTS_ALLOWED_INTERNAL_IMPORTS
                    )
                ):
                    add_violation(node, "agents-forbidden-internal-import", target)

                if is_agents and any(
                    _is_import_prefix(target, prefix)
                    for prefix in AGENTS_FORBIDDEN_EXTERNAL_IMPORTS
                ):
                    add_violation(node, "agents-forbidden-external-import", target)

                if any(
                    _is_import_prefix(target, prefix) for prefix in AGENT_FRAMEWORK_IMPORTS
                ) and not any(
                    _is_import_prefix(target, prefix)
                    and module in AGENT_FRAMEWORK_ALLOWED_MODULES[prefix]
                    for prefix in AGENT_FRAMEWORK_IMPORTS
                ):
                    add_violation(node, "agent-framework-import-outside-adapter", target)

                if is_api and any(
                    _is_import_prefix(target, prefix) for prefix in API_FORBIDDEN_INTERNAL_IMPORTS
                ):
                    add_violation(node, "api-forbidden-internal-import", target)

                if is_api and any(
                    _is_import_prefix(target, prefix) for prefix in API_FORBIDDEN_EXTERNAL_IMPORTS
                ):
                    add_violation(node, "api-forbidden-external-import", target)

                if is_domain and any(
                    _is_import_prefix(target, prefix)
                    for prefix in DOMAIN_FORBIDDEN_INTERNAL_IMPORTS
                ):
                    add_violation(node, "domain-forbidden-internal-import", target)

                if is_domain and any(
                    _is_import_prefix(target, prefix)
                    for prefix in DOMAIN_FORBIDDEN_EXTERNAL_IMPORTS
                ):
                    add_violation(node, "domain-forbidden-external-import", target)

                if (
                    is_db
                    and any(
                        _is_import_prefix(target, prefix)
                        for prefix in DB_FORBIDDEN_INTERNAL_IMPORTS
                    )
                    and not _is_import_prefix(target, "app.llm.invocations")
                    and not (
                        module == ("app", "db", "documents")
                        and _is_import_prefix(target, "app.retrieval.documents")
                    )
                ):
                    add_violation(node, "db-forbidden-internal-import", target)

                if is_db and any(
                    _is_import_prefix(target, prefix) for prefix in DB_FORBIDDEN_EXTERNAL_IMPORTS
                ):
                    add_violation(node, "db-forbidden-external-import", target)

                if is_llm and any(
                    _is_import_prefix(target, prefix) for prefix in LLM_FORBIDDEN_INTERNAL_IMPORTS
                ):
                    add_violation(node, "llm-forbidden-internal-import", target)

                if is_retrieval and any(
                    _is_import_prefix(target, prefix)
                    for prefix in RETRIEVAL_FORBIDDEN_INTERNAL_IMPORTS
                ):
                    add_violation(node, "retrieval-forbidden-internal-import", target)

                if is_retrieval and any(
                    _is_import_prefix(target, prefix)
                    for prefix in RETRIEVAL_FORBIDDEN_EXTERNAL_IMPORTS
                ):
                    add_violation(node, "retrieval-forbidden-external-import", target)

                if is_tools and any(
                    _is_import_prefix(target, prefix) for prefix in TOOLS_FORBIDDEN_INTERNAL_IMPORTS
                ):
                    add_violation(node, "tools-forbidden-internal-import", target)

                if (
                    is_worker
                    and module != ("app", "worker", "main")
                    and _is_import_prefix(target, "app.db")
                ):
                    add_violation(node, "worker-db-import-outside-composition-root", target)

                if is_worker and any(
                    _is_import_prefix(target, prefix)
                    for prefix in WORKER_FORBIDDEN_EXTERNAL_IMPORTS
                ):
                    add_violation(node, "worker-forbidden-external-import", target)

                if is_worker and _is_import_prefix(target, "app.mock_portal"):
                    add_violation(node, "worker-mock-portal-import", target)

                if is_mock_portal and any(
                    _is_import_prefix(target, prefix)
                    for prefix in MOCK_PORTAL_FORBIDDEN_INTERNAL_IMPORTS
                ):
                    add_violation(node, "mock-portal-forbidden-internal-import", target)

                if is_fake_provider and any(
                    _is_import_prefix(target, prefix) for prefix in FAKE_FORBIDDEN_IMPORTS
                ):
                    add_violation(node, "fake-provider-forbidden-import", target)

                if (
                    any(_is_import_prefix(target, prefix) for prefix in DATABASE_LIBRARY_IMPORTS)
                    and not is_db
                ):
                    add_violation(node, "database-library-import-outside-db", target)

                if any(_is_import_prefix(target, prefix) for prefix in OPENAI_IMPORTS) and not (
                    _is_module_within(module, ("app", "llm"))
                ):
                    add_violation(node, "openai-import-outside-llm", target)

                if any(_is_import_prefix(target, prefix) for prefix in JWT_IMPORTS) and not is_auth:
                    add_violation(node, "jwt-import-outside-auth", target)

                tavily_adapter = module == ("app", "tools", "adapters", "tavily")
                if any(_is_import_prefix(target, prefix) for prefix in TAVILY_IMPORTS) and not (
                    tavily_adapter
                ):
                    add_violation(node, "tavily-import-outside-adapter", target)

                if any(_is_import_prefix(target, prefix) for prefix in STRUCTLOG_IMPORTS) and not (
                    _is_module_within(module, ("app", "obs"))
                ):
                    add_violation(node, "structlog-import-outside-obs", target)

                if any(
                    _is_import_prefix(target, prefix) for prefix in PYDANTIC_SETTINGS_IMPORTS
                ) and module != ("app", "config"):
                    add_violation(node, "pydantic-settings-import-outside-config", target)

                if any(
                    _is_import_prefix(target, prefix) for prefix in UVICORN_IMPORTS
                ) and module != ("app", "main"):
                    add_violation(node, "uvicorn-import-outside-composition-root", target)

        if module != ("app", "main"):
            constructors, modules = _fastapi_bindings(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _is_fastapi_constructor(
                    node, constructors, modules
                ):
                    add_violation(
                        node,
                        "fastapi-instantiated-outside-composition-root",
                        ".".join(module),
                    )

    return sorted(violations)


def _write_source(source_root: Path, relative_path: str, source: str) -> None:
    path = source_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source).lstrip(), encoding="utf-8")


def test_project_source_respects_import_boundaries() -> None:
    scanned_paths = {
        path.relative_to(APP_SOURCE_ROOT).as_posix() for path in APP_SOURCE_ROOT.rglob("*.py")
    }

    assert {
        "__init__.py",
        "agents/contracts.py",
        "agents/create_agent_loop.py",
        "agents/prompting.py",
        "agents/prompts/__init__.py",
        "main.py",
        "agents/manual_loop.py",
        "api/router.py",
        "agents/research_contracts.py",
        "agents/research_graph.py",
        "api/errors.py",
        "api/middleware/request_context.py",
        "api/routes/events.py",
        "api/routes/health.py",
        "api/schemas/health.py",
        "api/schemas/problem.py",
        "cli/__init__.py",
        "cli/ingest_documents.py",
        "config.py",
        "db/base.py",
        "db/events.py",
        "db/migrations/env.py",
        "db/migrations/versions/0001_identity_baseline.py",
        "db/migrations/versions/0002_llm_invocations.py",
        "db/migrations/versions/0007_persistent_run_state.py",
        "db/llm_invocations.py",
        "db/jobs.py",
        "db/models.py",
        "db/provisioning.py",
        "db/readiness.py",
        "db/session.py",
        "domain/errors.py",
        "domain/jobs.py",
        "domain/provisioning.py",
        "domain/runs.py",
        "domain/tenancy.py",
        "domain/tracing.py",
        "domain/tool_effects.py",
        "domain/tool_invocations.py",
        "events/contracts.py",
        "llm/fake.py",
        "llm/factory.py",
        "llm/invocations.py",
        "llm/ports.py",
        "obs/logging.py",
        "obs/agent_loop.py",
        "obs/redaction.py",
        "retrieval/__init__.py",
        "retrieval/ingestion.py",
        "tools/contracts.py",
        "tools/fake_search.py",
        "tools/registry.py",
        "tools/search.py",
        "tools/teaching.py",
        "worker/contracts.py",
        "worker/fake_executor.py",
        "worker/main.py",
        "worker/runner.py",
        "worker/settings.py",
    } <= scanned_paths
    assert find_violations(APP_SOURCE_ROOT) == []


def test_event_query_paths_do_not_read_checkpoint_state() -> None:
    event_query_paths = (
        APP_SOURCE_ROOT / "api" / "routes" / "events.py",
        APP_SOURCE_ROOT / "db" / "events.py",
        APP_SOURCE_ROOT / "events" / "contracts.py",
    )
    forbidden_prefixes = (
        "app.db.checkpoints",
        "app.worker.checkpoints",
        "langgraph",
    )
    violations: list[str] = []
    for path in event_query_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module, is_package = _module_for_path(path, APP_SOURCE_ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            for target in _import_targets(module, is_package, node):
                if any(_is_import_prefix(target, prefix) for prefix in forbidden_prefixes):
                    violations.append(f"{path.relative_to(APP_SOURCE_ROOT)}: {target}")
    assert violations == []


def test_worker_system_ports_are_not_imported_by_product_layers() -> None:
    protected_names = {"JobClaimer", "StaleLeaseReclaimer", "WorkerJobStore"}
    allowed_roots = {"db", "domain", "worker"}
    violations: list[str] = []
    for path in APP_SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(APP_SOURCE_ROOT)
        if relative.parts[0] in allowed_roots:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "app.domain.jobs":
                continue
            imported = {alias.name for alias in node.names}
            if imported & protected_names:
                violations.append(relative.as_posix())
    assert violations == []


def test_production_agent_entry_does_not_import_the_manual_loop() -> None:
    adapter_path = APP_SOURCE_ROOT / "agents" / "create_agent_loop.py"
    tree = ast.parse(adapter_path.read_text(encoding="utf-8"), filename=str(adapter_path))
    module = ("app", "agents", "create_agent_loop")
    imported_targets = {
        target
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for target in _import_targets(module, False, node)
    }

    assert not any(
        _is_import_prefix(target, "app.agents.manual_loop") for target in imported_targets
    )


def test_checker_accepts_allowed_dependency_directions(tmp_path: Path) -> None:
    source_root = tmp_path / "app"
    _write_source(
        source_root,
        "main.py",
        """
        from fastapi import FastAPI
        import uvicorn
        application = FastAPI()
        """,
    )
    _write_source(source_root, "api/router.py", "from app.domain import runs\n")
    _write_source(source_root, "domain/runs.py", "from dataclasses import dataclass\n")
    _write_source(
        source_root,
        "agents/loop.py",
        "from app.llm import ports\nfrom app.tools import contracts\n",
    )
    _write_source(
        source_root,
        "agents/research_graph.py",
        "from langgraph.graph import StateGraph\n",
    )
    _write_source(source_root, "db/models.py", "from sqlalchemy.orm import DeclarativeBase\n")
    _write_source(source_root, "db/repository.py", "from app.domain import runs\n")
    _write_source(
        source_root,
        "db/llm_recorder.py",
        "from app.llm.invocations import InvocationRecorderPort\n",
    )
    _write_source(
        source_root,
        "db/documents.py",
        "from app.retrieval.documents import DocumentRepositoryPort\n",
    )
    _write_source(source_root, "llm/factory.py", "from openai import OpenAI\n")
    _write_source(source_root, "auth/supabase.py", "from jwt import PyJWKClient\n")
    _write_source(source_root, "retrieval/service.py", "from app.llm import factory\n")
    _write_source(source_root, "tools/adapters/tavily.py", "import tavily\n")
    _write_source(
        source_root,
        "tools/registry.py",
        "async def execute(spec, value, context):\n    return await spec.handler(value, context)\n",
    )
    _write_source(source_root, "obs/logging.py", "import structlog\n")
    _write_source(source_root, "obs/langfuse.py", "import langfuse\n")
    _write_source(
        source_root,
        "domain/tracing.py",
        "from __future__ import annotations\nfrom collections.abc import Mapping\n"
        "from dataclasses import dataclass\nfrom uuid import UUID\n",
    )
    _write_source(source_root, "config.py", "from pydantic_settings import BaseSettings\n")
    _write_source(source_root, "api/dbx.py", "import app.dbx\n")

    assert find_violations(source_root) == []


@pytest.mark.parametrize(
    ("relative_path", "source", "expected_rule", "expected_line"),
    [
        (
            "api/router.py",
            "from app import db\n",
            "api-forbidden-internal-import",
            1,
        ),
        (
            "agents/loop.py",
            "from app import db\n",
            "agents-forbidden-internal-import",
            1,
        ),
        (
            "agents/loop.py",
            "from app.worker import runner\n",
            "agents-forbidden-internal-import",
            1,
        ),
        (
            "agents/loop.py",
            "import httpx\n",
            "agents-forbidden-external-import",
            1,
        ),
        (
            "agents/manual_loop.py",
            "from langchain.agents import create_agent\n",
            "agent-framework-import-outside-adapter",
            1,
        ),
        (
            "agents/research_graph.py",
            "from langchain.agents import create_agent\n",
            "agent-framework-import-outside-adapter",
            1,
        ),
        (
            "agents/another_graph.py",
            "from langgraph.graph import StateGraph\n",
            "agent-framework-import-outside-adapter",
            1,
        ),
        (
            "tools/framework.py",
            "from langchain_core.tools import StructuredTool\n",
            "agent-framework-import-outside-adapter",
            1,
        ),
        (
            "domain/runs.py",
            "from fastapi import APIRouter\n",
            "domain-forbidden-external-import",
            1,
        ),
        (
            "api/provider.py",
            "from openai import OpenAI\n",
            "openai-import-outside-llm",
            1,
        ),
        (
            "api/auth.py",
            "import jwt\n",
            "jwt-import-outside-auth",
            1,
        ),
        (
            "api/application.py",
            """
            import fastapi as framework
            application = framework.FastAPI()
            """,
            "fastapi-instantiated-outside-composition-root",
            2,
        ),
        (
            "tools/other.py",
            "import tavily\n",
            "tavily-import-outside-adapter",
            1,
        ),
        (
            "tools/tavily_escape.py",
            "import tavily\n",
            "tavily-import-outside-adapter",
            1,
        ),
        (
            "api/logging.py",
            "import structlog\n",
            "structlog-import-outside-obs",
            1,
        ),
        (
            "domain/settings.py",
            "from pydantic_settings import BaseSettings\n",
            "pydantic-settings-import-outside-config",
            1,
        ),
        (
            "obs/database.py",
            "import sqlalchemy\n",
            "database-library-import-outside-db",
            1,
        ),
        (
            "retrieval/vector.py",
            "from pgvector.sqlalchemy import Vector\n",
            "database-library-import-outside-db",
            1,
        ),
        (
            "db/other_retrieval.py",
            "from app.retrieval.documents import DocumentRepositoryPort\n",
            "db-forbidden-internal-import",
            1,
        ),
        (
            "db/http.py",
            "from fastapi import Request\n",
            "db-forbidden-external-import",
            1,
        ),
        (
            "db/router.py",
            "from app.api import router\n",
            "db-forbidden-internal-import",
            1,
        ),
        (
            "worker/executor.py",
            "from app.mock_portal import service\n",
            "worker-mock-portal-import",
            1,
        ),
        (
            "worker/executor.py",
            "from app.db import mock_submissions\n",
            "worker-db-import-outside-composition-root",
            1,
        ),
        (
            "mock_portal/service.py",
            "from app.worker import runner\n",
            "mock-portal-forbidden-internal-import",
            1,
        ),
        (
            "db/provider.py",
            "from app.llm.factory import LLMFactory\n",
            "db-forbidden-internal-import",
            1,
        ),
        (
            "api/server.py",
            "import uvicorn\n",
            "uvicorn-import-outside-composition-root",
            1,
        ),
        (
            "llm/fake.py",
            "from app.db import models\n",
            "llm-forbidden-internal-import",
            1,
        ),
        (
            "retrieval/service.py",
            "from app.db import models\n",
            "retrieval-forbidden-internal-import",
            1,
        ),
        (
            "retrieval/service.py",
            "import sqlalchemy\n",
            "retrieval-forbidden-external-import",
            1,
        ),
        (
            "tools/fake_search.py",
            "from app.worker import runner\n",
            "tools-forbidden-internal-import",
            1,
        ),
        (
            "llm/fake.py",
            "import socket\n",
            "fake-provider-forbidden-import",
            1,
        ),
        (
            "llm/fake.py",
            "from openai import OpenAI\n",
            "fake-provider-forbidden-import",
            1,
        ),
        (
            "tools/fake_search.py",
            "import os\n",
            "fake-provider-forbidden-import",
            1,
        ),
        (
            "agents/loop.py",
            "async def bypass(spec, value, context):\n"
            "    return await spec.handler(value, context)\n",
            "tool-handler-called-outside-registry",
            2,
        ),
    ],
)
def test_checker_reports_injected_violations(
    tmp_path: Path,
    relative_path: str,
    source: str,
    expected_rule: str,
    expected_line: int,
) -> None:
    source_root = tmp_path / "app"
    _write_source(source_root, relative_path, source)

    violation = next(item for item in find_violations(source_root) if item.rule == expected_rule)

    assert violation.relative_path == relative_path
    assert violation.lineno == expected_line
    assert violation.detail


def test_checker_resolves_relative_imports(tmp_path: Path) -> None:
    source_root = tmp_path / "app"
    _write_source(source_root, "api/router.py", "from ..db import Session\n")
    _write_source(source_root, "api/routes/health.py", "from ...db import Repository\n")
    _write_source(source_root, "agents/loop.py", "from ..db import Repository\n")

    violations = find_violations(source_root)

    assert {
        (violation.relative_path, violation.detail)
        for violation in violations
        if violation.rule == "api-forbidden-internal-import"
    } == {
        ("api/router.py", "app.db"),
        ("api/routes/health.py", "app.db"),
    }
    assert {
        (violation.relative_path, violation.detail)
        for violation in violations
        if violation.rule == "agents-forbidden-internal-import"
    } == {
        ("agents/loop.py", "app.db"),
    }


def test_checker_rejects_missing_source_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="application source root is not a directory"):
        find_violations(tmp_path / "missing")


@pytest.mark.parametrize("source", ["import mcp", "from mcp.server import MCPServer"])
def test_checker_allows_mcp_only_in_experiment(tmp_path: Path, source: str) -> None:
    source_root = tmp_path / "app"
    _write_source(source_root, "tools/mcp_experiment/server.py", source)
    assert find_violations(source_root) == []


@pytest.mark.parametrize("path", ["worker/foo.py", "agents/foo.py", "tools/ordinary.py", "main.py"])
@pytest.mark.parametrize("source", ["import mcp", "from mcp import Client", "import mcp_types"])
def test_checker_rejects_mcp_outside_experiment(tmp_path: Path, path: str, source: str) -> None:
    source_root = tmp_path / "app"
    _write_source(source_root, path, source)
    assert any(
        item.rule == "mcp-import-outside-experiment" for item in find_violations(source_root)
    )


@pytest.mark.parametrize("path", ["main.py", "worker/main.py", "agents/loop.py", "tools/other.py"])
@pytest.mark.parametrize(
    "source",
    [
        "from app.tools.mcp_experiment.client import open_mcp_experiment",
        "import app.tools.mcp_experiment.client",
        "from app.tools import mcp_experiment",
    ],
)
def test_checker_rejects_indirect_mcp_dependency(tmp_path: Path, path: str, source: str) -> None:
    source_root = tmp_path / "app"
    _write_source(source_root, path, source)
    assert any(
        item.rule == "experiment-import-from-production" for item in find_violations(source_root)
    )


def test_checker_allows_internal_experiment_import(tmp_path: Path) -> None:
    source_root = tmp_path / "app"
    _write_source(source_root, "tools/mcp_experiment/client.py", "from .server import SERVER_NAME")
    assert find_violations(source_root) == []


def test_checker_rejects_relative_experiment_import(tmp_path: Path) -> None:
    source_root = tmp_path / "app"
    _write_source(source_root, "tools/other.py", "from .mcp_experiment import client")
    assert any(
        item.rule == "experiment-import-from-production" for item in find_violations(source_root)
    )


def test_checker_rejects_bare_except(tmp_path: Path) -> None:
    source_root = tmp_path / "app"
    _write_source(
        source_root,
        "domain/service.py",
        """
        try:
            raise RuntimeError
        except:
            pass
        """,
    )

    violations = find_violations(source_root)

    violation = next(item for item in violations if item.rule == "bare-except")
    assert violation.relative_path == "domain/service.py"
    assert violation.lineno == 3


@pytest.mark.parametrize("layer", ["worker", "agents", "domain", "tools"])
@pytest.mark.parametrize("source", ["import langfuse", "from langfuse import Langfuse"])
def test_checker_rejects_langfuse_outside_obs(tmp_path: Path, layer: str, source: str) -> None:
    source_root = tmp_path / "app"
    _write_source(source_root, f"{layer}/tracing.py", source)
    assert any(item.rule == "langfuse-import-outside-obs" for item in find_violations(source_root))


@pytest.mark.parametrize(
    "source",
    [
        "import app.obs",
        "import app.llm",
        "import app.tools",
        "import app.agents",
        "import app.worker",
        "import app.retrieval",
        "import langfuse",
        "import langgraph",
        "import langchain",
        "import fastapi",
        "import sqlalchemy",
        "import pydantic",
        "import httpx",
        "from ..llm import ports",
        "from app import llm",
        "from pydantic import BaseModel",
    ],
)
def test_checker_rejects_non_stdlib_tracing_imports(tmp_path: Path, source: str) -> None:
    source_root = tmp_path / "app"
    _write_source(source_root, "domain/tracing.py", source)
    assert any(
        item.rule == "domain-tracing-non-stdlib-import" for item in find_violations(source_root)
    )


@pytest.mark.parametrize("layer", ["tools", "retrieval"])
@pytest.mark.parametrize("source", ["import app.obs", "from ..obs import langfuse"])
def test_checker_rejects_business_exporter_imports(tmp_path: Path, layer: str, source: str) -> None:
    source_root = tmp_path / "app"
    _write_source(source_root, f"{layer}/tracing.py", source)
    assert any(
        item.rule == f"{layer}-forbidden-internal-import" for item in find_violations(source_root)
    )
