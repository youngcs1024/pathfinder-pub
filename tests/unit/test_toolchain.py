from __future__ import annotations

import platform
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PYTHON_VERSION = "3.12.13"


def test_runtime_uses_locked_cpython() -> None:
    assert platform.python_implementation() == "CPython"
    assert sys.version_info[:3] == (3, 12, 13)


def test_python_version_file_matches_locked_runtime() -> None:
    version_file = PROJECT_ROOT / ".python-version"

    assert version_file.read_text(encoding="utf-8") == f"{EXPECTED_PYTHON_VERSION}\n"


def test_project_metadata_requires_exact_python() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == f"=={EXPECTED_PYTHON_VERSION}"
