from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from tests.evals.harness import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MANIFEST_PATH,
    TRUSTED_CONTEXT_CANARY,
    V1_DATASET_PATH,
    EvalConfigurationError,
    _research_script,
    load_eval_dataset,
    load_eval_manifest,
    runtime_version_metadata,
)


def _first_valid_payload() -> dict[str, object]:
    first_line = DEFAULT_DATASET_PATH.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(first_line)


def _payload_by_id(case_id: str) -> dict[str, Any]:
    return next(
        json.loads(line)
        for line in DEFAULT_DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["case_id"] == case_id
    )


def _write_jsonl(path: Path, *payloads: object) -> None:
    path.write_text(
        "".join(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            for payload in payloads
        ),
        encoding="utf-8",
    )


def test_default_dataset_is_strict_versioned_and_covers_web_and_document_cases() -> None:
    cases = load_eval_dataset()

    assert tuple(case.case_id for case in cases) == (
        "normal_application",
        "insufficient",
        "conflicting_sources",
        "prompt_injection",
        "duplicate_url",
        "document_only_resume",
        "web_and_resume",
        "resume_irrelevant",
        "resume_prompt_injection",
        "no_document_scope",
        "application_resume_draft",
        "reserved_tool_context_injection",
        "unknown_tool_proposal",
        "tool_budget_exhausted",
    )
    assert all(case.schema_version == 2 for case in cases)
    assert all(case.expectations.max_input_tokens >= 0 for case in cases)
    assert (
        sha256(V1_DATASET_PATH.read_bytes()).hexdigest()
        == "26e681516f30c9918595bedb32ced48552fb249f2d9783ca42d3c2f1b06e8901"
    )
    assert all(
        case.expectations.max_input_tokens <= case.expectations.max_model_calls
        and case.expectations.max_output_tokens <= case.expectations.max_model_calls
        for case in cases
    )


def test_negative_tool_proposal_fixture_is_strict_and_omits_unused_completion() -> None:
    case = next(
        item for item in load_eval_dataset() if item.case_id == "reserved_tool_context_injection"
    )

    assert case.research_passes[0].expected_rejection == "invalid_tool_arguments"
    assert case.research_passes[0].ordered_tool_calls[1].model_extra_arguments == {
        "workspace_id": "forged-model-context"
    }
    script = _research_script(case)
    assert len(script) == 3
    assert len(script[0].tool_calls) == 2
    assert script[1].tool_calls[0].arguments == {
        "query": "synthetic platform engineer role",
        "max_results": 3,
    }
    assert script[2].content == "Bounded offline research pass complete."


@pytest.mark.parametrize("canonical_name", ("query", "max_results"))
def test_model_extra_arguments_cannot_override_canonical_arguments(
    tmp_path: Path, canonical_name: str
) -> None:
    payload = _payload_by_id("reserved_tool_context_injection")
    payload["research_passes"][0]["ordered_tool_calls"][1]["model_extra_arguments"] = {
        canonical_name: "forged"
    }
    path = tmp_path / "canonical-override.jsonl"
    _write_jsonl(path, payload)

    with pytest.raises(EvalConfigurationError, match="invalid_dataset"):
        load_eval_dataset(path)


def test_negative_arguments_require_explicit_rejection(tmp_path: Path) -> None:
    payload = _payload_by_id("reserved_tool_context_injection")
    payload["research_passes"][0].pop("expected_rejection")
    path = tmp_path / "unmarked-negative.jsonl"
    _write_jsonl(path, payload)

    with pytest.raises(EvalConfigurationError, match="invalid_dataset"):
        load_eval_dataset(path)


def test_rejected_proposal_requires_negative_ordered_arguments(tmp_path: Path) -> None:
    payload = _payload_by_id("reserved_tool_context_injection")
    payload["research_passes"][0]["ordered_tool_calls"][1].pop("model_extra_arguments")
    path = tmp_path / "missing-negative.jsonl"
    _write_jsonl(path, payload)

    with pytest.raises(EvalConfigurationError, match="invalid_dataset"):
        load_eval_dataset(path)


def test_v3_manifest_matches_code_owned_profiles_and_hashes() -> None:
    assert load_eval_manifest() == runtime_version_metadata()


def test_v3_manifest_fails_closed_when_a_prompt_hash_drifts(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["plan_prompt_version"] = "sha256:" + "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvalConfigurationError) as captured:
        load_eval_manifest(path)

    assert captured.value.category == "invalid_manifest"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"schema_version": 1}),
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload["support_rules"][0].update(
            {"allowed_evidence_aliases": ["missing_evidence"]}
        ),
        lambda payload: payload["research_passes"][0]["calls"][0].update(
            {"query": "missing search fixture"}
        ),
    ],
    ids=["unknown-version", "extra-field", "invalid-support", "unknown-search"],
)
def test_dataset_rejects_invalid_contract_or_references(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    payload = _first_valid_payload()
    mutation(payload)
    path = tmp_path / "invalid.jsonl"
    _write_jsonl(path, payload)

    with pytest.raises(EvalConfigurationError) as captured:
        load_eval_dataset(path)

    assert captured.value.category == "invalid_dataset"


@pytest.mark.parametrize("content", ["{\n", "\n", "\xff"])
def test_dataset_rejects_malformed_blank_or_non_utf8_content(
    tmp_path: Path,
    content: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    if content == "\xff":
        path.write_bytes(b"\xff")
    else:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(EvalConfigurationError):
        load_eval_dataset(path)


def test_dataset_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    payload = _first_valid_payload()
    path = tmp_path / "duplicate.jsonl"
    _write_jsonl(path, payload, payload)

    with pytest.raises(EvalConfigurationError) as captured:
        load_eval_dataset(path)

    assert captured.value.category == "invalid_dataset"


def test_dataset_cannot_contain_the_trusted_context_canary(tmp_path: Path) -> None:
    payload = _first_valid_payload()
    payload["request"]["query"] = TRUSTED_CONTEXT_CANARY  # type: ignore[index]
    path = tmp_path / "canary.jsonl"
    _write_jsonl(path, payload)

    with pytest.raises(EvalConfigurationError) as captured:
        load_eval_dataset(path)

    assert captured.value.category == "invalid_dataset"


def test_unavailable_dataset_has_a_distinct_safe_category(tmp_path: Path) -> None:
    with pytest.raises(EvalConfigurationError) as captured:
        load_eval_dataset(tmp_path / "missing.jsonl")

    assert captured.value.category == "dataset_unavailable"
