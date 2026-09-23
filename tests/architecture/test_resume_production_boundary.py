import ast
from pathlib import Path

from app.main import create_app

ROOT = Path(__file__).resolve().parents[2]


def test_production_roots_do_not_wire_retired_execution_or_test_assemblies():
    prohibited = (
        "tests",
        "app.mock_portal",
        "app.worker.langgraph_executor",
        "app.worker.fake_research_adapter",
        "app.tools.adapters.mock_portal",
        "app.tools.adapters.tavily",
        "app.tools.mock_application",
        "app.db.approval_expiry",
    )
    for name in ("main.py", "worker/main.py"):
        tree = ast.parse((ROOT / "src/app" / name).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
        assert not any(
            module == banned or module.startswith(banned + ".")
            for module in imports
            for banned in prohibited
        )


def test_only_material_writes_and_explicitly_retired_legacy_routes_are_published():
    app = create_app()
    writes = [
        (path, spec["post"]) for path, spec in app.openapi()["paths"].items() if "post" in spec
    ]
    assert {path for path, _ in writes} == {
        "/api/v1/workspaces/{workspace_id}/runs",
        "/api/v1/workspaces/{workspace_id}/runs/{run_id}/cancel",
        "/api/v1/workspaces/{workspace_id}/action-intents/{action_intent_id}/decision",
        "/api/v2/workspaces/{workspace_id}/projects",
        "/api/v2/workspaces/{workspace_id}/projects/{project_id}/material-sources",
        "/api/v2/workspaces/{workspace_id}/projects/{project_id}/imports",
    }
    assert all(
        "410" in spec["responses"]
        and "200" not in spec["responses"]
        and "202" not in spec["responses"]
        for path, spec in writes
        if path.startswith("/api/v1/")
    )
    assert all(
        "410" not in spec["responses"] for path, spec in writes if path.startswith("/api/v2/")
    )
