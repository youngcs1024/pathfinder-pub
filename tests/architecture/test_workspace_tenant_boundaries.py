from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_SOURCE_ROOT = PROJECT_ROOT / "src" / "app"

# Product-facing workspace-owned query methods. Worker-only bounded CAS ports are
# intentionally absent: JobClaimer, StaleLeaseReclaimer, and
# ApprovalRequestExpirySweeper are the documented narrow system exceptions.
WORKSPACE_QUERY_SURFACES = {
    "db/runs.py": {"get_run": "tenant"},
    "db/events.py": {"read_after": "tenant"},
    "db/documents.py": {"find_complete": "tenant", "search": "tenant"},
    "db/approvals.py": {"get_action_review": "tenant"},
}

TYPED_SCOPE_CARRIERS = {
    "db/actions.py": {"prepare_action": "command"},
    "db/approvals.py": {"decide": "command"},
    "db/llm_invocations.py": {"prepare": "attempt"},
}


def _method_arguments(node: ast.AsyncFunctionDef) -> set[str]:
    arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    return {argument.arg for argument in arguments if argument.arg != "self"}


def find_unscoped_workspace_queries(
    source: str,
    required_methods: dict[str, str],
) -> list[str]:
    tree = ast.parse(dedent(source))
    found: dict[str, ast.AsyncFunctionDef] = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
    }
    violations: list[str] = []
    for method_name, scope_argument in required_methods.items():
        method = found.get(method_name)
        if method is None:
            violations.append(f"{method_name}: missing audited method")
        elif scope_argument not in _method_arguments(method):
            violations.append(f"{method_name}: missing {scope_argument} scope")
    return violations


def test_workspace_owned_product_queries_expose_explicit_tenant_scope() -> None:
    violations: list[str] = []
    for relative_path, methods in WORKSPACE_QUERY_SURFACES.items():
        source = (APP_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        violations.extend(
            f"{relative_path}: {violation}"
            for violation in find_unscoped_workspace_queries(source, methods)
        )
    assert violations == []


def test_reviewed_commands_and_attempts_keep_their_typed_scope_carriers() -> None:
    violations: list[str] = []
    for relative_path, methods in TYPED_SCOPE_CARRIERS.items():
        source = (APP_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        violations.extend(
            f"{relative_path}: {violation}"
            for violation in find_unscoped_workspace_queries(source, methods)
        )
    assert violations == []


def test_checker_rejects_unscoped_query_and_accepts_tenant_context() -> None:
    unsafe = """
    class UnsafeRunRepository:
        async def get_run(self, run_id):
            return run_id
    """
    safe = """
    class SafeRunRepository:
        async def get_run(self, *, tenant: TenantContext, run_id):
            return tenant, run_id
    """

    assert find_unscoped_workspace_queries(unsafe, {"get_run": "tenant"}) == [
        "get_run: missing tenant scope"
    ]
    assert find_unscoped_workspace_queries(safe, {"get_run": "tenant"}) == []
