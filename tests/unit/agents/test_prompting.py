from __future__ import annotations

import re
from importlib import resources
from typing import cast

import pytest

from app.agents.prompting import (
    AgentPromptBundleError,
    AgentPromptProfile,
    _load_prompt_bundle,
    _load_research_plan_prompt,
    _load_research_writer_prompt,
    load_agent_prompt_bundle,
    load_research_plan_prompt,
    load_research_writer_prompt,
)

EXPECTED_PROMPT_VERSION = "sha256:ea8834aed78c1ddd7f33ccb02bb540e692ee5fb826729a243f0eaa6395c744b5"
EXPECTED_RESEARCH_PROMPT_VERSION = (
    "sha256:14b811368f69bc6fb67ed28dd4f0ca2c087a099e725d0cc71c06e9beda73101a"
)
EXPECTED_PLAN_PROMPT_VERSION = (
    "sha256:d6fcfd84d71bf65a1b634e635912d32fd8b6d7db246f655de444e2397f01010a"
)
EXPECTED_WRITER_PROMPT_VERSION = (
    "sha256:ef97580f25d257aa12d03d95224cebc9ccc407bab7500a1885083f844f7501c7"
)
PROMPT_FILES = ("system.md", "tool.md", "synthesis.md")


def _packaged_resources() -> dict[str, bytes]:
    prompt_root = resources.files("app.agents.prompts")
    return {name: prompt_root.joinpath(name).read_bytes() for name in PROMPT_FILES}


def test_packaged_prompt_bundle_has_fixed_order_policy_and_snapshot() -> None:
    observed_names: list[str] = []
    packaged = _packaged_resources()

    def read_resource(name: str) -> bytes:
        observed_names.append(name)
        return packaged[name]

    bundle = _load_prompt_bundle(read_resource)

    assert observed_names == list(PROMPT_FILES)
    assert bundle.version == EXPECTED_PROMPT_VERSION
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", bundle.version)
    assert bundle.system_prompt.isascii()
    assert bundle.system_prompt.index("# System policy") < bundle.system_prompt.index(
        "# Tool-use policy"
    )
    assert bundle.system_prompt.index("# Tool-use policy") < bundle.system_prompt.index(
        "# Synthesis policy"
    )
    for required_policy in (
        "code-owned policy",
        "untrusted data",
        "statically declared tools",
        "reserved execution fields",
        "conform exactly",
        "non-empty plain-text final",
        "do not fabricate evidence",
        "hidden reasoning",
    ):
        assert required_policy in bundle.system_prompt
    assert load_agent_prompt_bundle() == bundle


@pytest.mark.parametrize(
    ("replacement", "expected_message"),
    [
        (b"", "agent prompt bundle contains an empty resource"),
        (b" \n\t", "agent prompt bundle contains an empty resource"),
        (b"\xff", "agent prompt bundle must use strict UTF-8"),
    ],
    ids=["empty", "whitespace", "invalid-utf8"],
)
def test_prompt_bundle_rejects_invalid_resources(
    replacement: bytes,
    expected_message: str,
) -> None:
    packaged = _packaged_resources()
    packaged["tool.md"] = replacement

    with pytest.raises(AgentPromptBundleError, match=f"^{expected_message}$"):
        _load_prompt_bundle(packaged.__getitem__)


def test_prompt_bundle_hides_missing_resource_details() -> None:
    canary = "missing-prompt-path-canary"

    def missing_resource(_name: str) -> bytes:
        raise FileNotFoundError(canary)

    with pytest.raises(
        AgentPromptBundleError,
        match=r"^agent prompt bundle is unavailable$",
    ) as captured:
        _load_prompt_bundle(missing_resource)
    assert canary not in str(captured.value)


def test_one_byte_prompt_change_changes_version() -> None:
    packaged = _packaged_resources()
    original = _load_prompt_bundle(packaged.__getitem__)
    packaged["synthesis.md"] += b"!"

    changed = _load_prompt_bundle(packaged.__getitem__)

    assert changed.system_prompt != original.system_prompt
    assert changed.version != original.version


def test_research_profile_is_fixed_and_adds_only_code_owned_policy() -> None:
    bundle = load_agent_prompt_bundle("research")

    assert bundle.version == EXPECTED_RESEARCH_PROMPT_VERSION
    assert bundle.system_prompt.startswith(load_agent_prompt_bundle().system_prompt)
    assert "# Research-stage policy" in bundle.system_prompt
    assert "actual Registry" in bundle.system_prompt
    assert "trusted runtime tool-budget notice" in bundle.system_prompt
    assert "Never propose more tool calls" in bundle.system_prompt
    assert "remaining tool budget is zero" in bundle.system_prompt


def test_arbitrary_agent_prompt_profile_is_rejected() -> None:
    with pytest.raises(AgentPromptBundleError, match=r"^agent prompt profile is invalid$"):
        load_agent_prompt_bundle(cast(AgentPromptProfile, "caller-controlled"))


def test_research_plan_prompt_is_fixed_versioned_and_strict() -> None:
    bundle = load_research_plan_prompt()

    assert bundle.version == EXPECTED_PLAN_PROMPT_VERSION
    assert "Return exactly one JSON object" in bundle.system_prompt
    assert "Do not use Markdown fences" in bundle.system_prompt
    assert "Instruction-shaped text" in bundle.system_prompt
    assert "reason to\nrefuse" in bundle.system_prompt
    assert _load_research_plan_prompt() == bundle


def test_research_plan_prompt_change_updates_version() -> None:
    packaged = resources.files("app.agents.prompts").joinpath("research_plan.md").read_bytes()
    original = _load_research_plan_prompt(lambda _name: packaged)
    changed = _load_research_plan_prompt(lambda _name: packaged + b"!")

    assert changed.system_prompt != original.system_prompt
    assert changed.version != original.version


def test_research_writer_prompt_is_fixed_versioned_and_strict() -> None:
    bundle = load_research_writer_prompt()

    assert bundle.version == EXPECTED_WRITER_PROMPT_VERSION
    assert "Return exactly one JSON object" in bundle.system_prompt
    assert "Every factual claim" in bundle.system_prompt
    assert "globally unique across `summary`, `findings`" in bundle.system_prompt
    assert "the draft is required" in bundle.system_prompt
    assert "concise bounded report" in bundle.system_prompt
    assert "at most 6 findings" in bundle.system_prompt
    assert "Prefer complete valid JSON over longer prose" in bundle.system_prompt
    assert "summary`, `findings`, `limitations`, and `application_draft`" in bundle.system_prompt
    assert "may contain only `claim_id`, `text`, and `citations`" in bundle.system_prompt
    assert "may contain only `source_id` and `evidence_id`" in bundle.system_prompt
    assert "Even when the supplied evidence has `source_type` set to" in bundle.system_prompt
    assert "never include `source_type`, `document_id`, `chunk_id`" in bundle.system_prompt
    assert "workspace-document-v1:00000000-0000-0000-0000-000000000000" in (bundle.system_prompt)
    assert "Pathfinder trusted application code derives the source type" in (bundle.system_prompt)
    assert "When `evidence_sufficient` is false" in bundle.system_prompt
    assert "never return `insufficient_evidence`" in bundle.system_prompt
    assert "only\nwhen the supplied evidence genuinely conflicts" in bundle.system_prompt
    assert "Writer must not reproduce or add" in bundle.system_prompt
    assert "resembles system, user, or tool instructions" in bundle.system_prompt
    assert "reason to refuse" in bundle.system_prompt
    assert "produce at least one supported summary or finding claim" in bundle.system_prompt
    assert "Do not use Markdown fences" in bundle.system_prompt
    assert _load_research_writer_prompt() == bundle


def test_research_writer_prompt_change_updates_version() -> None:
    packaged = resources.files("app.agents.prompts").joinpath("research_writer.md").read_bytes()
    original = _load_research_writer_prompt(lambda _name: packaged)
    changed = _load_research_writer_prompt(lambda _name: packaged + b"!")

    assert changed.system_prompt != original.system_prompt
    assert changed.version != original.version


@pytest.mark.parametrize(
    ("replacement", "expected_message"),
    [
        (b"", "research writer prompt is empty"),
        (b"\xff", "research writer prompt must use strict UTF-8"),
    ],
    ids=["empty", "invalid-utf8"],
)
def test_research_writer_prompt_rejects_invalid_resource(
    replacement: bytes,
    expected_message: str,
) -> None:
    with pytest.raises(AgentPromptBundleError, match=f"^{expected_message}$"):
        _load_research_writer_prompt(lambda _name: replacement)


def test_research_writer_prompt_hides_missing_resource_details() -> None:
    canary = "missing-writer-prompt-canary"

    def missing_resource(_name: str) -> bytes:
        raise FileNotFoundError(canary)

    with pytest.raises(
        AgentPromptBundleError,
        match=r"^research writer prompt is unavailable$",
    ) as captured:
        _load_research_writer_prompt(missing_resource)
    assert canary not in str(captured.value)
