from __future__ import annotations

import pytest

import tests.evals.harness as harness
from tests.evals.grader_vectors import GRADER_BEHAVIOR_VECTORS, GraderBehaviorVector
from tests.evals.harness import grade_eval_case


def _actual(vector: GraderBehaviorVector) -> tuple[tuple[str, bool, int], ...]:
    report = grade_eval_case(
        vector.case,
        vector.output,
        vector.search_calls,
        model_call_count=vector.model_call_count,
        evidence_alias_by_id=vector.evidence_alias_by_id,
        document_retrieval_calls=vector.document_retrieval_calls,
        relevant_chunk_ids=vector.relevant_chunk_ids,
        irrelevant_chunk_ids=vector.irrelevant_chunk_ids,
        input_tokens=vector.input_tokens,
        output_tokens=vector.output_tokens,
        tool_invocation_reservation_count=vector.tool_invocation_reservation_count,
    )
    return tuple((grader.name, grader.passed, grader.failure_count) for grader in report.graders)


@pytest.mark.parametrize(
    "vector",
    GRADER_BEHAVIOR_VECTORS,
    ids=lambda vector: vector.vector_id,
)
def test_grader_behavior_vector(vector: GraderBehaviorVector) -> None:
    assert _actual(vector) == vector.expected


def test_vectors_detect_a_temporarily_weakened_grader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = next(item for item in GRADER_BEHAVIOR_VECTORS if item.vector_id == "unsupported_claim")
    monkeypatch.setattr(harness, "_unsupported_claim_ids", lambda *args: ())

    with pytest.raises(AssertionError):
        assert _actual(vector) == vector.expected
