"""R3.2 keeps rendering and delivery inside their existing module boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _imports(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text())
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
    return result


def test_renderer_and_artifact_contract_have_no_runtime_or_database_dependencies() -> None:
    for path in (
        "src/app/resume/template_render.py",
        "src/app/domain/resume_artifacts.py",
    ):
        assert not any(
            name.startswith(("app.api", "app.db", "app.llm", "app.worker", "sqlalchemy", "fastapi"))
            for name in _imports(path)
        )


def test_artifact_download_route_does_not_render_or_query_orm_directly() -> None:
    imports = _imports("src/app/api/routes/resume_artifacts.py")
    assert not any(
        name.startswith(("app.db", "app.resume", "app.agents", "app.worker")) for name in imports
    )
