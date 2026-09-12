from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from tests.evals.retrieval_benchmark import (
    DEFAULT_RETRIEVAL_DATASET_PATH,
    RetrievalBenchmarkConfigurationError,
    load_retrieval_dataset,
    normalized_retrieval_report,
    recall_at,
    reciprocal_rank,
    retrieval_dataset_digests,
    run_retrieval_benchmark,
)
from tests.evals.retrieval_contracts import (
    RetrievalBenchmarkCaseReportV1,
    RetrievalBenchmarkReportV1,
    RetrievalChunkRefV1,
    RetrievalDatasetV1,
)


def _payload() -> dict[str, object]:
    return json.loads(DEFAULT_RETRIEVAL_DATASET_PATH.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_dataset_schema_and_stable_identity_are_valid() -> None:
    dataset = load_retrieval_dataset()
    dataset_digest, case_set_digest = retrieval_dataset_digests(dataset)

    assert dataset.dataset_version == "retrieval-v1"
    assert len(dataset.documents) == 5
    assert len(dataset.cases) == 10
    assert dataset_digest.startswith("sha256:")
    assert case_set_digest.startswith("sha256:")
    assert len({case.case_id for case in dataset.cases}) == 10
    assert sum(document.workspace == "foreign" for document in dataset.documents) == 1


@pytest.mark.parametrize("mutation", ["malformed", "duplicate_alias", "duplicate_case"])
def test_malformed_and_duplicate_datasets_fail_closed(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "retrieval.json"
    if mutation == "malformed":
        path.write_text("{", encoding="utf-8")
    else:
        payload = _payload()
        if mutation == "duplicate_alias":
            payload["documents"][1]["alias"] = payload["documents"][0]["alias"]  # type: ignore[index]
        else:
            payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]  # type: ignore[index]
        _write(path, payload)

    with pytest.raises(RetrievalBenchmarkConfigurationError):
        load_retrieval_dataset(path)


def test_unresolved_document_alias_fails_closed() -> None:
    payload = _payload()
    payload["cases"][0]["relevant_chunks"][0]["document_alias"] = "missing"  # type: ignore[index]

    with pytest.raises(ValidationError, match="unresolved"):
        RetrievalDatasetV1.model_validate_json(json.dumps(payload), strict=True)


def test_metric_definitions_use_actual_ranked_ids() -> None:
    first = UUID("00000000-0000-0000-0000-000000000001")
    second = UUID("00000000-0000-0000-0000-000000000002")
    third = UUID("00000000-0000-0000-0000-000000000003")
    ranked = (first, second, third)
    relevant = frozenset({second, third})

    assert recall_at(ranked, relevant, 1) == 0
    assert recall_at(ranked, relevant, 3) == 1
    assert recall_at(ranked, relevant, 5) == 1
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert reciprocal_rank((first,), relevant) == 0


def test_report_contract_labels_fake_embedding_evidence_as_non_semantic() -> None:
    fields = RetrievalBenchmarkReportV1.model_fields
    assert fields["evidence_scope"].annotation is not None
    assert fields["semantic_quality_claim"].default is False


@pytest.mark.parametrize(
    ("passed", "error_category"),
    ((False, None), (True, "retrieval_invariant_failed")),
)
def test_case_report_status_and_error_category_must_agree(
    passed: bool,
    error_category: str | None,
) -> None:
    reference = RetrievalChunkRefV1(document_alias="backend_profile", ordinal=0)
    with pytest.raises(ValidationError, match="status"):
        RetrievalBenchmarkCaseReportV1(
            case_id="example",
            relevant_chunks=(reference,),
            ranked_retrieved_chunks=(reference,),
            recall_at_1=1,
            recall_at_3=1,
            recall_at_5=1,
            reciprocal_rank=1,
            irrelevant_context_count=0,
            workspace_leakage_count=0,
            allowlist_leakage_count=0,
            passed=passed,
            error_category=error_category,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_benchmark_requires_explicit_disposable_database_confirmation() -> None:
    def sessions() -> object:
        raise AssertionError("database must not be queried without explicit confirmation")

    with pytest.raises(RetrievalBenchmarkConfigurationError, match="explicit disposable"):
        await run_retrieval_benchmark(sessions)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_nonmatching_database_name_fails_before_any_write() -> None:
    class Session:
        async def scalar(self, statement: object) -> str:
            assert str(statement) == "SELECT current_database()"
            return "pathfinder_production"

    class SessionContext:
        async def __aenter__(self) -> Session:
            return Session()

        async def __aexit__(self, *args: object) -> None:
            return None

    class Sessions:
        call_count = 0

        def __call__(self) -> SessionContext:
            self.call_count += 1
            if self.call_count > 1:
                raise AssertionError("benchmark attempted a write-capable session")
            return SessionContext()

    sessions = Sessions()
    with pytest.raises(RetrievalBenchmarkConfigurationError, match="isolated"):
        await run_retrieval_benchmark(  # type: ignore[arg-type]
            sessions,
            confirm_disposable_database=True,
        )
    assert sessions.call_count == 1


def test_normalized_report_is_canonical() -> None:
    payload = {
        "schema_version": 1,
        "evidence_scope": "deterministic_fake_embedding_regression",
        "semantic_quality_claim": False,
        "dataset_version": "retrieval-v1",
        "dataset_digest": "sha256:" + "1" * 64,
        "case_set_digest": "sha256:" + "2" * 64,
        "normalization_version": "nfc-lf-v1",
        "chunking_version": "heading-paragraph-utf8-budget-800-v1",
        "embedding_profile": "qwen-beijing-text-embedding-v4-1536-v1",
        "embedding_dimension": 1536,
        "retrieval_top_k": 5,
        "cases": [
            {
                "case_id": "example",
                "relevant_chunks": [{"document_alias": "backend_profile", "ordinal": 0}],
                "ranked_retrieved_chunks": [],
                "recall_at_1": 0,
                "recall_at_3": 0,
                "recall_at_5": 0,
                "reciprocal_rank": 0,
                "irrelevant_context_count": 0,
                "workspace_leakage_count": 0,
                "allowlist_leakage_count": 0,
                "passed": True,
                "error_category": None,
            }
        ],
        "aggregate": {
            "macro_recall_at_1": 0,
            "macro_recall_at_3": 0,
            "macro_recall_at_5": 0,
            "mean_reciprocal_rank": 0,
            "total_irrelevant_context_count": 0,
            "workspace_leakage_count": 0,
            "allowlist_leakage_count": 0,
        },
        "passed": True,
    }
    report = RetrievalBenchmarkReportV1.model_validate_json(json.dumps(payload), strict=True)
    first = normalized_retrieval_report(report)
    second = normalized_retrieval_report(
        RetrievalBenchmarkReportV1.model_validate_json(first, strict=True)
    )

    assert first == second
    assert first.endswith(b"\n")
