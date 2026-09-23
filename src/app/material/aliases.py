from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import UUID

from app.domain.errors import DomainValidationError

_ALIAS = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def safe_relative_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or str(PurePosixPath(value)) != value
    ):
        raise DomainValidationError("material path is invalid")
    return value


@dataclass(frozen=True, slots=True)
class MaterialAlias:
    name: str
    kind: Literal["git", "file"]
    root: Path
    paths: tuple[str, ...]
    workspace_ids: tuple[UUID, ...]
    commit: str | None = None

    def __post_init__(self) -> None:
        if (
            _ALIAS.fullmatch(self.name) is None
            or self.kind not in {"git", "file"}
            or not self.root.is_absolute()
            or not self.paths
            or not self.workspace_ids
            or len(set(self.workspace_ids)) != len(self.workspace_ids)
            or len(set(self.paths)) != len(self.paths)
            or (self.kind == "git") != (self.commit is not None)
            or (self.commit is not None and _COMMIT.fullmatch(self.commit) is None)
        ):
            raise DomainValidationError("material alias is invalid")
        for path in self.paths:
            safe_relative_path(path)

    @property
    def digest(self) -> str:
        canonical = json.dumps(
            {
                "name": self.name,
                "kind": self.kind,
                "root": str(self.root),
                "paths": self.paths,
                "workspace_ids": tuple(str(item) for item in self.workspace_ids),
                "commit": self.commit,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class MaterialAliasRegistry:
    aliases: tuple[MaterialAlias, ...] = ()

    def __post_init__(self) -> None:
        if len({alias.name for alias in self.aliases}) != len(self.aliases):
            raise DomainValidationError("material aliases must be unique")

    def get(self, name: str, workspace_id: UUID) -> MaterialAlias:
        for alias in self.aliases:
            if alias.name == name and workspace_id in alias.workspace_ids:
                return alias
        raise DomainValidationError("material alias is unavailable")


def load_aliases(path: Path | None) -> MaterialAliasRegistry:
    if path is None:
        return MaterialAliasRegistry()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"aliases"} or not isinstance(raw["aliases"], list):
        raise DomainValidationError("material alias configuration is invalid")
    aliases: list[MaterialAlias] = []
    for item in raw["aliases"]:
        if not isinstance(item, dict) or set(item) - {
            "name",
            "kind",
            "root",
            "paths",
            "workspace_ids",
            "commit",
        }:
            raise DomainValidationError("material alias configuration is invalid")
        if not isinstance(item.get("workspace_ids"), list) or not item["workspace_ids"]:
            raise DomainValidationError("material alias configuration is invalid")
        if not isinstance(item.get("paths"), list) or not all(
            isinstance(value, str) for value in item["paths"]
        ):
            raise DomainValidationError("material alias configuration is invalid")
        aliases.append(
            MaterialAlias(
                name=item["name"],
                kind=item["kind"],
                root=Path(item["root"]),
                paths=tuple(item["paths"]),
                workspace_ids=tuple(UUID(value) for value in item["workspace_ids"]),
                commit=item.get("commit"),
            )
        )
    return MaterialAliasRegistry(tuple(aliases))
