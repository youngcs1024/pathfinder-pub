from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from typing import Literal

_ASSEMBLY_ID = b"pathfinder-agent-prompt-bundle-v1"
_RESEARCH_ASSEMBLY_ID = b"pathfinder-research-agent-prompt-bundle-v3"
_PLAN_ASSEMBLY_ID = b"pathfinder-research-plan-prompt-v2"
_WRITER_ASSEMBLY_ID = b"pathfinder-research-writer-prompt-v7"
_BASE_PROMPT_FILES = ("system.md", "tool.md", "synthesis.md")
_RESEARCH_PROMPT_FILES = (*_BASE_PROMPT_FILES, "research.md")
_PLAN_PROMPT_FILE = "research_plan.md"
_WRITER_PROMPT_FILE = "research_writer.md"

AgentPromptProfile = Literal["base", "research"]


class AgentPromptBundleError(Exception):
    """The code-owned Agent prompt bundle is missing or invalid."""


@dataclass(frozen=True, slots=True)
class AgentPromptBundle:
    system_prompt: str
    version: str


type PromptResourceReader = Callable[[str], bytes]


def _read_packaged_prompt(name: str) -> bytes:
    try:
        return resources.files("app.agents.prompts").joinpath(name).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        raise AgentPromptBundleError("agent prompt bundle is unavailable") from None


def _load_prompt_bundle(
    read_resource: PromptResourceReader = _read_packaged_prompt,
    *,
    profile: AgentPromptProfile = "base",
) -> AgentPromptBundle:
    if profile == "base":
        assembly_id = _ASSEMBLY_ID
        prompt_files = _BASE_PROMPT_FILES
    elif profile == "research":
        assembly_id = _RESEARCH_ASSEMBLY_ID
        prompt_files = _RESEARCH_PROMPT_FILES
    else:
        raise AgentPromptBundleError("agent prompt profile is invalid")

    digest = sha256()
    digest.update(assembly_id)
    sections: list[str] = []

    for name in prompt_files:
        try:
            raw_content = read_resource(name)
        except AgentPromptBundleError:
            raise
        except Exception:
            raise AgentPromptBundleError("agent prompt bundle is unavailable") from None
        if not isinstance(raw_content, bytes):
            raise AgentPromptBundleError("agent prompt bundle contains an invalid resource")
        try:
            content = raw_content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise AgentPromptBundleError("agent prompt bundle must use strict UTF-8") from None
        if not content.strip():
            raise AgentPromptBundleError("agent prompt bundle contains an empty resource")

        encoded_name = name.encode("utf-8")
        digest.update(encoded_name)
        digest.update(len(raw_content).to_bytes(8, byteorder="big", signed=False))
        digest.update(raw_content)
        sections.append(content)

    return AgentPromptBundle(
        system_prompt="\n\n".join(sections),
        version=f"sha256:{digest.hexdigest()}",
    )


def load_agent_prompt_bundle(
    profile: AgentPromptProfile = "base",
) -> AgentPromptBundle:
    """Load and version the immutable packaged prompt bundle on demand."""

    return _load_prompt_bundle(profile=profile)


def _load_standalone_prompt(
    *,
    assembly_id: bytes,
    prompt_file: str,
    resource_label: str,
    read_resource: PromptResourceReader,
) -> AgentPromptBundle:
    try:
        raw_content = read_resource(prompt_file)
    except AgentPromptBundleError:
        raise
    except Exception:
        raise AgentPromptBundleError(f"{resource_label} prompt is unavailable") from None
    if not isinstance(raw_content, bytes):
        raise AgentPromptBundleError(f"{resource_label} prompt contains an invalid resource")
    try:
        content = raw_content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise AgentPromptBundleError(f"{resource_label} prompt must use strict UTF-8") from None
    if not content.strip():
        raise AgentPromptBundleError(f"{resource_label} prompt is empty")

    digest = sha256()
    digest.update(assembly_id)
    digest.update(prompt_file.encode("utf-8"))
    digest.update(len(raw_content).to_bytes(8, byteorder="big", signed=False))
    digest.update(raw_content)
    return AgentPromptBundle(
        system_prompt=content,
        version=f"sha256:{digest.hexdigest()}",
    )


def _load_research_plan_prompt(
    read_resource: PromptResourceReader = _read_packaged_prompt,
) -> AgentPromptBundle:
    return _load_standalone_prompt(
        assembly_id=_PLAN_ASSEMBLY_ID,
        prompt_file=_PLAN_PROMPT_FILE,
        resource_label="research plan",
        read_resource=read_resource,
    )


def load_research_plan_prompt() -> AgentPromptBundle:
    """Load the code-owned one-shot research planning prompt."""

    return _load_research_plan_prompt()


def _load_research_writer_prompt(
    read_resource: PromptResourceReader = _read_packaged_prompt,
) -> AgentPromptBundle:
    return _load_standalone_prompt(
        assembly_id=_WRITER_ASSEMBLY_ID,
        prompt_file=_WRITER_PROMPT_FILE,
        resource_label="research writer",
        read_resource=read_resource,
    )


def load_research_writer_prompt() -> AgentPromptBundle:
    """Load the code-owned one-shot research writer prompt."""

    return _load_research_writer_prompt()
