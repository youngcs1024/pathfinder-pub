"""E4.6 private publication and observation helpers; never a production entrypoint."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from tests.evals.contracts import TRUSTED_CONTEXT_CANARY
from tests.evals.live_chat import FOREIGN_WORKSPACE_CANARY, SECRET_CANARY
from tests.evals.quality_contracts import (
    QualityGenerationCaseV1,
    QualityGenerationReportV1,
    QualityGenerationStartV1,
    QualityPrivateCaseV1,
    QualityPrivateFileV1,
    QualityPrivateOutputV1,
    QualityPrivateSourcesV1,
    QualityRetrievalRepresentationV1,
    QualitySafetyCountsV1,
)
from tests.evals.quality_dataset import quality_digest


class QualityGenerationError(Exception):
    """Fixed, content-free error; underlying exceptions never cross this boundary."""


class GenerationSafetyError(Exception):
    pass


@dataclass(repr=False)
class GenerationGuard:
    markers: tuple[str, ...] = ()
    secret_leak: int = 0
    unauthorized_access: int = 0
    integrity_failed: bool = False
    configuration_failed: bool = False
    tool_accounting_complete: bool = True
    tool_failed: bool = False

    def check(self) -> None:
        if self.secret_leak or self.unauthorized_access:
            raise GenerationSafetyError
        if self.integrity_failed or self.configuration_failed or not self.tool_accounting_complete:
            raise QualityGenerationError("generation_integrity_stopped")

    def scan(self, content: str, *, system_prompts: tuple[str, ...] = ()) -> None:
        if any(marker and marker in content for marker in self.markers) or any(
            prompt and prompt in content for prompt in system_prompts
        ):
            self.secret_leak += 1
            raise GenerationSafetyError from None

    def scope_failure(self) -> None:
        self.unauthorized_access += 1
        raise GenerationSafetyError

    def safety(self) -> QualitySafetyCountsV1:
        return QualitySafetyCountsV1(
            secret_leak=self.secret_leak, unauthorized_access=self.unauthorized_access
        )


def secret_markers(additional: tuple[str, ...] = ()) -> tuple[str, ...]:
    # Read only for local leak detection, never provider selection or artifact metadata.
    values = tuple(
        value
        for key, value in os.environ.items()
        if any(part in key.upper() for part in ("API_KEY", "TOKEN", "SECRET", "DATABASE_URL"))
        and len(value) >= 8
    )
    return (SECRET_CANARY, TRUSTED_CONTEXT_CANARY, FOREIGN_WORKSPACE_CANARY, *values, *additional)


def checked_directory(path: Path, *, private: bool = True) -> Path:
    absolute = Path(os.path.abspath(path))
    for parent in (*reversed(absolute.parents), absolute):
        if parent.is_symlink():
            raise QualityGenerationError("unsafe_artifact_directory")
    info = absolute.stat()
    if not stat.S_ISDIR(info.st_mode) or (
        private and (stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid())
    ):
        raise QualityGenerationError("unsafe_artifact_directory")
    return absolute


PUBLIC_TYPES = (
    QualityGenerationStartV1,
    QualityGenerationCaseV1,
    QualityGenerationReportV1,
    QualityRetrievalRepresentationV1,
)
PRIVATE_TYPES = (
    QualityGenerationStartV1,
    QualityPrivateSourcesV1,
    QualityPrivateCaseV1,
    QualityPrivateOutputV1,
    QualityGenerationReportV1,
)


def artifact_bytes(payload, *, private: bool) -> bytes:
    if type(payload) not in (PRIVATE_TYPES if private else PUBLIC_TYPES):
        raise QualityGenerationError("unsupported_artifact_contract")
    checked = type(payload).model_validate_json(payload.model_dump_json())
    return (
        json.dumps(
            checked.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(repr=False)
class GenerationArtifacts:
    public_dir: Path
    private_dir: Path
    guard: GenerationGuard
    files: list[QualityPrivateFileV1] = field(default_factory=list)

    @classmethod
    def reserve(cls, public_dir: Path, private_root: Path, experiment: str, guard, repo: Path):
        try:
            root = checked_directory(private_root)
            public_parent = checked_directory(public_dir.parent, private=False)
            public = public_parent / public_dir.name
            private = root / experiment
            if (
                not experiment
                or Path(experiment).name != experiment
                or experiment in {".", ".."}
                or root.is_relative_to(repo.resolve())
                or repo.resolve().is_relative_to(root)
                or public.is_relative_to(root)
                or root.is_relative_to(public)
                or public.name in {"", ".", ".."}
                or private.exists()
                or private.is_symlink()
                or public.exists()
                or public.is_symlink()
            ):
                raise QualityGenerationError("unsafe_artifact_directory")
            # Reserve the private identity first. Any partial directory is deliberately retained.
            private.mkdir(mode=0o700, exist_ok=False)
            public.mkdir(mode=0o700, exist_ok=False)
            return cls(public, private, guard)
        except Exception:
            raise QualityGenerationError("artifact_reservation_failed") from None

    def write(self, name: str, payload, *, private: bool = False) -> str:
        try:
            if Path(name).name != name or name in {"", ".", ".."}:
                raise QualityGenerationError("invalid_artifact_name")
            data = artifact_bytes(payload, private=private)
            self.guard.scan(data.decode("utf-8"))
            directory = checked_directory(self.private_dir if private else self.public_dir)
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                fd = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=descriptor,
                )
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            digest = quality_digest(data)
            if private and name != "report.json":
                self.files.append(
                    QualityPrivateFileV1(name=name, digest=digest, byte_count=len(data))
                )
            return digest
        except GenerationSafetyError:
            raise
        except Exception:
            raise QualityGenerationError("artifact_publication_failed") from None
