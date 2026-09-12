"""Bounded CI diagnostics: never publish exception bodies or parameter values."""

import re
from pathlib import Path


def collection_diagnostic(output: str, *, root: Path) -> str:
    bounded = output[:262144]
    category = (
        "import_error"
        if any(t in bounded for t in ("ImportError", "ModuleNotFoundError"))
        else "collection_error"
    )
    files = []
    for candidate in re.findall(r"ERROR collecting ([a-zA-Z0-9_./-]+\.py)", bounded):
        path = Path(candidate)
        if path.is_absolute() or ".." in path.parts or not candidate.startswith("tests/"):
            continue
        resolved = (root / path).resolve()
        if (
            resolved.is_relative_to(root.resolve())
            and resolved.is_file()
            and candidate not in files
        ):
            files.append(candidate)
        if len(files) == 8:
            break
    return "category=" + category + (" files=" + ",".join(files) if files else "")
