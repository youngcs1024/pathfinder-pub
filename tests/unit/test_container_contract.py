import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
ALEMBIC_CONFIG = PROJECT_ROOT / "alembic.ini"
_FROM_PATTERN = re.compile(r"^FROM\s+(?P<image>\S+)(?:\s+AS\s+\S+)?$", re.MULTILINE)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def test_application_image_uses_locked_noneditable_production_install() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM ghcr.io/astral-sh/uv:0.11.32@sha256:" in dockerfile
    assert dockerfile.count("FROM python:3.12.13-slim-bookworm@sha256:") == 1
    assert "COPY pyproject.toml uv.lock .python-version ./" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "uv sync" in dockerfile
    assert "--locked" in dockerfile
    assert "--no-dev" in dockerfile
    assert "--no-editable" in dockerfile
    assert "COPY . ." not in dockerfile
    for business_secret in (
        "DASHSCOPE_API_KEY",
        "TAVILY_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    ):
        assert business_secret not in dockerfile


def test_application_image_carries_package_relative_alembic_configuration() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    alembic_config = ALEMBIC_CONFIG.read_text(encoding="utf-8")

    assert "COPY alembic.ini /app/alembic.ini" in dockerfile
    assert "script_location = app.db:migrations" in alembic_config
    assert "src/app/db/migrations" not in alembic_config


def test_every_external_base_is_digest_pinned_and_python_stages_match() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    images = [match.group("image") for match in _FROM_PATTERN.finditer(dockerfile)]

    assert len(images) == 4
    assert images[-2:] == ["python-base", "python-base"]
    images = images[:2]
    references: list[tuple[str, str]] = []
    for image in images:
        reference, separator, digest = image.rpartition("@sha256:")
        assert separator == "@sha256:"
        assert _SHA256_PATTERN.fullmatch(digest) is not None
        assert ":" in reference
        references.append((reference, digest))

    uv_references = [item for item in references if item[0].startswith("ghcr.io/astral-sh/uv:")]
    python_references = [item for item in references if item[0].startswith("python:")]
    assert uv_references == [
        (
            "ghcr.io/astral-sh/uv:0.11.32",
            "df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c",
        )
    ]
    assert len(python_references) == 1
    assert {reference for reference, _digest in python_references} == {
        "python:3.12.13-slim-bookworm"
    }
    assert len({digest for _reference, digest in python_references}) == 1


def test_final_image_runs_as_fixed_nonroot_with_tmp_home() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    final_stage = dockerfile.rsplit("\nFROM ", maxsplit=1)[1]

    assert "USER 10001:10001" in final_stage
    assert "HOME=/tmp" in final_stage
    assert "TMPDIR=/tmp" in final_stage
    assert "PYTHONDONTWRITEBYTECODE=1" in final_stage
    assert "PYTHONUNBUFFERED=1" in final_stage
    assert "chmod" not in final_stage.lower()


def test_docker_context_excludes_local_and_nonproduction_content() -> None:
    ignored = set(DOCKERIGNORE.read_text(encoding="utf-8").splitlines())

    assert {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        ".env",
        ".env.*",
        "tests",
        "docs",
        "evals",
    } <= ignored


def test_security_fix_is_exact_and_shared():
    body = DOCKERFILE.read_text()
    assert "AS python-base" in body
    assert "FROM python-base AS builder" in body
    assert "FROM python-base AS runtime" in body
    assert "--only-upgrade libpcre2-8-0=10.42-1+deb12u1" in body
    assert "dpkg-query -W" in body
    assert "type=tmpfs,target=/var/lib/apt/lists" in body
    assert "apt-get upgrade" not in body
