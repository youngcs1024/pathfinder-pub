"""R3.1 keeps private source parsing and profile rules out of model execution."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _imports(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text())
    names = set()
    for item in ast.walk(tree):
        if isinstance(item, ast.ImportFrom) and item.module:
            names.add(item.module)
        elif isinstance(item, ast.Import):
            names.update(alias.name for alias in item.names)
    return names


def test_resume_profile_domain_and_parser_do_not_depend_on_runtime_or_orm() -> None:
    for path in (
        "src/app/domain/resume_profile.py",
        "src/app/domain/resume_profiles.py",
        "src/app/resume/template_import.py",
    ):
        names = _imports(path)
        assert not any(
            name.startswith(
                ("app.api", "app.db", "app.llm", "app.worker", "sqlalchemy", "langgraph", "fastapi")
            )
            for name in names
        )


def test_resume_profile_api_has_no_model_or_worker_execution_imports() -> None:
    names = _imports("src/app/api/routes/resume_profiles.py")
    assert not any(
        name.startswith(("app.db", "app.agents", "app.llm", "app.worker")) for name in names
    )
    names = _imports("src/app/db/resume_profiles.py")
    assert not any(name.startswith(("app.agents", "app.llm", "app.worker")) for name in names)
