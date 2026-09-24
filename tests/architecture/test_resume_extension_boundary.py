"""R6.1 leaves optional capabilities out of production registration."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_production_has_no_test_adapter_imports_or_optional_registration():
    for path in (ROOT / "src/app").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("tests")
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("tests") for alias in node.names)
    for name in ("src/app/main.py", "src/app/worker/main.py"):
        tree = ast.parse((ROOT / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not {keyword.arg for keyword in node.keywords} & {
                    "template_manifest",
                    "template_source",
                    "source_reader",
                }
    for name in ("src/app/domain/job_inputs.py", "src/app/domain/resume_templates.py"):
        tree = ast.parse((ROOT / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    (
                        "app.db",
                        "app.api",
                        "app.worker",
                        "app.llm",
                        "app.resume",
                        "httpx",
                        "subprocess",
                    )
                )
