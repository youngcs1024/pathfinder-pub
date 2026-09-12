from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import tests.evals.harness as harness
from app.domain.runs import CURRENT_GRAPH_VERSION
from app.retrieval.documents import EMBEDDING_PROFILE
from tests.evals.contracts import (
    EVAL_GRADER_CONTRACT_VERSION,
    EVAL_GRADER_NAMES,
    EvalReportV3,
)
from tests.evals.grader_vectors import (
    GRADER_BEHAVIOR_VECTORS,
    grader_expected_results_payload,
)
from tests.evals.harness import (
    DEFAULT_DATASET_PATH,
    TRUSTED_CONTEXT_CANARY,
    eval_artifact_identity,
    eval_dataset_digests,
    load_eval_dataset,
    run_evaluation,
)


def _write_cases(path: Path, payloads: list[dict[str, Any]], *, pretty: bool = False) -> None:
    if pretty:
        lines = [json.dumps(payload, ensure_ascii=False, sort_keys=False) for payload in payloads]
    else:
        lines = [
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) for payload in payloads
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def test_full_and_selected_artifacts_share_dataset_identity_and_define_scope() -> None:
    full_report, full_exit = await run_evaluation()
    selected_report, selected_exit = await run_evaluation(selected_case="normal_application")

    assert full_exit == selected_exit == 0
    assert full_report.schema_version == selected_report.schema_version == 3
    assert full_report.passed is selected_report.passed is True
    assert len(full_report.cases) == 14
    assert full_report.artifact_identity == selected_report.artifact_identity
    assert full_report.artifact_identity is not None
    assert full_report.artifact_identity.graph_version == CURRENT_GRAPH_VERSION
    assert full_report.artifact_identity.embedding_profile == EMBEDDING_PROFILE
    assert full_report.artifact_identity.grader_contract_version == EVAL_GRADER_CONTRACT_VERSION
    assert full_report.artifact_identity.grader_contract_version == "research-graders-v3"
    assert full_report.comparison_metadata is not None
    assert selected_report.comparison_metadata is not None
    assert full_report.comparison_metadata.scope == "full_dataset"
    assert (
        full_report.comparison_metadata.dataset_case_ids
        == full_report.comparison_metadata.evaluated_case_ids
    )
    assert selected_report.comparison_metadata.scope == "selected_case"
    assert selected_report.comparison_metadata.evaluated_case_ids == ("normal_application",)
    assert (
        selected_report.comparison_metadata.dataset_case_ids
        == full_report.comparison_metadata.dataset_case_ids
    )


def test_dataset_identity_is_path_and_json_serialization_independent(tmp_path: Path) -> None:
    payloads = [
        json.loads(line) for line in DEFAULT_DATASET_PATH.read_text(encoding="utf-8").splitlines()
    ]
    compact_path = tmp_path / "first" / "dataset.jsonl"
    pretty_path = tmp_path / "second" / "renamed.jsonl"
    compact_path.parent.mkdir()
    pretty_path.parent.mkdir()
    _write_cases(compact_path, payloads)
    reordered = [dict(reversed(payload.items())) for payload in reversed(payloads)]
    _write_cases(pretty_path, reordered, pretty=True)

    compact_cases = load_eval_dataset(compact_path)
    pretty_cases = load_eval_dataset(pretty_path)

    assert eval_dataset_digests(compact_cases) == eval_dataset_digests(pretty_cases)
    assert eval_dataset_digests(compact_cases) == eval_dataset_digests(load_eval_dataset())


def test_dataset_content_and_case_set_drift_have_distinct_identities() -> None:
    cases = load_eval_dataset()
    original_dataset_digest, original_case_set_digest = eval_dataset_digests(cases)
    first = cases[0]
    changed_first = first.model_copy(
        update={
            "expectations": first.expectations.model_copy(
                update={"minimum_cited_sources": first.expectations.minimum_cited_sources + 1}
            )
        }
    )
    changed_cases = (changed_first, *cases[1:])
    changed_dataset_digest, changed_case_set_digest = eval_dataset_digests(changed_cases)
    removed_dataset_digest, removed_case_set_digest = eval_dataset_digests(cases[:-1])

    assert changed_dataset_digest != original_dataset_digest
    assert changed_case_set_digest == original_case_set_digest
    assert removed_case_set_digest != original_case_set_digest
    assert removed_dataset_digest != original_dataset_digest


async def test_grader_identity_and_aggregates_match_per_case_evidence() -> None:
    report, exit_code = await run_evaluation()

    assert exit_code == 0
    assert report.artifact_identity == eval_artifact_identity(load_eval_dataset())
    assert tuple(item.name for item in report.grader_aggregates) == EVAL_GRADER_NAMES
    for aggregate in report.grader_aggregates:
        per_case = [
            next(grader for grader in case.graders if grader.name == aggregate.name)
            for case in report.cases
        ]
        assert aggregate.passed_case_count == sum(grader.passed for grader in per_case)
        assert aggregate.failed_case_count == sum(not grader.passed for grader in per_case)
        assert aggregate.failure_count == sum(grader.failure_count for grader in per_case)


def test_grader_expected_results_payload_is_sorted_and_behavior_sensitive() -> None:
    payload = grader_expected_results_payload()
    assert [item["vector_id"] for item in payload] == sorted(
        vector.vector_id for vector in GRADER_BEHAVIOR_VECTORS
    )

    all_pass = next(vector for vector in GRADER_BEHAVIOR_VECTORS if vector.vector_id == "all_pass")
    weakened = replace(
        all_pass,
        expected=tuple(
            (name, False, 1) if name == "unsupported_claim" else item
            for item in all_pass.expected
            for name in (item[0],)
        ),
    )
    changed = grader_expected_results_payload(
        tuple(
            weakened if vector.vector_id == "all_pass" else vector
            for vector in reversed(GRADER_BEHAVIOR_VECTORS)
        )
    )
    assert changed != payload


def test_grader_behavior_payload_changes_artifact_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = load_eval_dataset()
    original = eval_artifact_identity(cases)
    payload = grader_expected_results_payload()
    changed_payload = [*payload]
    changed_payload[0] = {
        **changed_payload[0],
        "expected_results": [
            {
                **changed_payload[0]["expected_results"][0],
                "passed": False,
                "failure_count": 1,
            },
            *changed_payload[0]["expected_results"][1:],
        ],
    }
    monkeypatch.setattr(harness, "grader_expected_results_payload", lambda: changed_payload)

    changed = eval_artifact_identity(cases)

    assert changed.dataset_digest == original.dataset_digest
    assert changed.case_set_digest == original.case_set_digest
    assert changed.grader_contract_digest != original.grader_contract_digest


async def test_document_retrieval_trace_is_preserved_without_bodies_or_context() -> None:
    cases = {case.case_id: case for case in load_eval_dataset()}
    report, exit_code = await run_evaluation(selected_case="document_only_resume")

    assert exit_code == 0
    case_report = report.cases[0]
    assert len(case_report.document_retrieval_calls) == 1
    trace = case_report.document_retrieval_calls[0]
    fixture_results = cases["document_only_resume"].document_retrievals[0].results
    assert trace.result_count == len(fixture_results)
    assert trace.document_ids == tuple(item.document_id for item in fixture_results)
    assert trace.chunk_ids == tuple(item.chunk_id for item in fixture_results)
    serialized_trace = trace.model_dump_json()
    assert all(item.untrusted_text not in serialized_trace for item in fixture_results)
    assert "query" not in serialized_trace
    assert TRUSTED_CONTEXT_CANARY not in serialized_trace

    no_scope_report, no_scope_exit = await run_evaluation(selected_case="no_document_scope")
    assert no_scope_exit == 0
    assert no_scope_report.cases[0].document_retrieval_calls == ()


async def test_configuration_failures_do_not_claim_valid_artifact_evidence(
    tmp_path: Path,
) -> None:
    malformed_dataset = tmp_path / "malformed.jsonl"
    malformed_dataset.write_text("{\n", encoding="utf-8")
    duplicate_dataset = tmp_path / "duplicate.jsonl"
    duplicate_payload = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8").splitlines()[0])
    _write_cases(duplicate_dataset, [duplicate_payload, duplicate_payload])
    invalid_manifest = tmp_path / "manifest.json"
    invalid_manifest.write_text("{}", encoding="utf-8")

    results = (
        await run_evaluation(dataset_path=tmp_path / "missing.jsonl"),
        await run_evaluation(dataset_path=malformed_dataset),
        await run_evaluation(dataset_path=duplicate_dataset),
        await run_evaluation(manifest_path=invalid_manifest),
        await run_evaluation(selected_case="unknown_case"),
        await run_evaluation(manifest_path=tmp_path / "missing-manifest.json"),
    )

    for (report, exit_code), expected_category in zip(
        results,
        (
            "dataset_unavailable",
            "invalid_dataset",
            "invalid_dataset",
            "invalid_manifest",
            "unknown_case",
            "manifest_unavailable",
        ),
        strict=True,
    ):
        assert exit_code == 2
        assert report.error_category == expected_category
        assert report.artifact_identity is None
        assert report.comparison_metadata is None
        assert report.grader_aggregates == ()
        assert report.cases == ()


async def test_normalized_artifact_excludes_nondeterministic_runtime_metadata() -> None:
    report, exit_code = await run_evaluation()
    assert exit_code == 0
    serialized = report.model_dump_json()
    for canary in (
        TRUSTED_CONTEXT_CANARY,
        "/tmp/pathfinder-eval-temporary-absolute-path-canary",
        "pathfinder-hostname-canary",
        "pathfinder-credential-canary",
        "pathfinder-dynamic-port-49152-canary",
    ):
        assert canary not in serialized

    payload = json.loads(serialized)
    forbidden_runtime_fields = {
        "generated_at",
        "hostname",
        "machine_name",
        "temporary_path",
        "dynamic_port",
        "process_id",
        "runtime_invocation_id",
        "duration",
        "duration_seconds",
    }

    def field_names(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {name for item in value.values() for name in field_names(item)}
        if isinstance(value, list):
            return {name for item in value for name in field_names(item)}
        return set()

    assert field_names(payload).isdisjoint(forbidden_runtime_fields)
    assert EvalReportV3.model_validate_json(serialized, strict=True) == report


async def test_valid_quality_failure_keeps_complete_artifact_evidence(tmp_path: Path) -> None:
    original_case = load_eval_dataset()[0]
    original_identity = eval_artifact_identity((original_case,))
    payload = original_case.model_dump(mode="json", round_trip=True)
    payload["expectations"]["minimum_cited_sources"] = 3
    dataset_path = tmp_path / "quality-failure.jsonl"
    _write_cases(dataset_path, [payload])

    report, exit_code = await run_evaluation(dataset_path=dataset_path)

    assert exit_code == 1
    assert report.passed is False
    assert report.error_category is None
    assert report.artifact_identity is not None
    assert report.artifact_identity.dataset_digest != original_identity.dataset_digest
    assert report.artifact_identity.case_set_digest == original_identity.case_set_digest
    assert report.comparison_metadata is not None
    assert report.grader_aggregates
    assert report.version_metadata is not None
    assert report.cases[0].passed is False
    assert (
        next(
            item for item in report.grader_aggregates if item.name == "source_diversity"
        ).failed_case_count
        == 1
    )
