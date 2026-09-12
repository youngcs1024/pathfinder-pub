from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.retrieval.documents import RETRIEVAL_TOP_K
from tests.evals.live_retrieval_contracts import (
    PRIMARY_ALIASES,
    AcceptedLiveRetrievalBaselineV2,
    LiveRetrievalAggregateV1,
    LiveRetrievalCaseReportV1,
    LiveRetrievalDatasetV1,
    LiveRetrievalPolicyV1,
    LiveRetrievalReportV1,
    LiveRetrievalReportV2,
    live_retrieval_digests,
    live_retrieval_passes,
    load_live_retrieval_dataset,
    load_live_retrieval_policy,
    prepared_live_documents,
)


def _case_report(query):  # type: ignore[no-untyped-def]
    ranked = query.relevant_chunks[:5]
    return LiveRetrievalCaseReportV1(
        query_id=query.query_id,
        relevant_chunks=query.relevant_chunks,
        ranked_retrieved_chunks=ranked,
        recall_at_1=min(1, 1 / len(query.relevant_chunks)),
        recall_at_3=1,
        recall_at_5=1,
        reciprocal_rank=1,
        irrelevant_context_count=0,
        workspace_leakage_count=0,
        allowlist_leakage_count=0,
        returned_context_bytes=100,
    )


def _report_payload():
    dataset, policy = load_live_retrieval_dataset(), load_live_retrieval_policy()
    cases = tuple(_case_report(query) for query in dataset.queries)
    count = len(cases)
    aggregate = LiveRetrievalAggregateV1(
        macro_recall_at_1=sum(case.recall_at_1 for case in cases) / count,
        macro_recall_at_3=1,
        macro_recall_at_5=1,
        mean_reciprocal_rank=1,
        total_irrelevant_context_count=0,
        workspace_leakage_count=0,
        allowlist_leakage_count=0,
    )
    return {
        "schema_version": 1,
        "evidence_scope": "exploratory_live_embedding_retrieval_benchmark",
        "exploratory": True,
        "provider": "qwen",
        "semantic_quality_claim": True,
        "dataset_version": policy.dataset_version,
        "dataset_digest": policy.dataset_digest,
        "case_set_digest": policy.case_set_digest,
        "normalization_version": policy.normalization_version,
        "chunking_version": policy.chunking_version,
        "embedding_profile": policy.embedding_profile,
        "embedding_dimension": policy.embedding_dimension,
        "retrieval_top_k": policy.retrieval_top_k,
        "candidate_pool_size": policy.expected_primary_candidate_pool_size,
        "query_count": policy.query_count,
        "complete": True,
        "cases": [case.model_dump(mode="json") for case in cases],
        "aggregate": aggregate.model_dump(mode="json"),
        "provider_attempt_count": 28,
        "input_tokens": 0,
        "known_cost_cny": "0",
        "unknown_cost_attempt_count": 0,
        "observed_embedding_attempt_latency_ms": [0] * 28,
        "passed": live_retrieval_passes(aggregate, policy, complete=True),
        "error_category": None,
    }


def test_strict_retrieval_inputs_and_identity_roundtrip():
    dataset, policy = load_live_retrieval_dataset(), load_live_retrieval_policy()
    copy = LiveRetrievalDatasetV1.model_validate_json(dataset.model_dump_json(), strict=True)
    assert dataset == copy
    assert (
        LiveRetrievalPolicyV1.model_validate_json(policy.model_dump_json(), strict=True) == policy
    )
    assert live_retrieval_digests(dataset) == live_retrieval_digests(copy)
    assert live_retrieval_digests(dataset) == (policy.dataset_digest, policy.case_set_digest)
    assert (
        policy.thresholds.minimum_macro_recall_at_1,
        policy.thresholds.minimum_macro_recall_at_3,
        policy.thresholds.minimum_macro_recall_at_5,
        policy.thresholds.minimum_mean_reciprocal_rank,
    ) == (0.50, 0.75, 0.85, 0.60)
    assert (
        policy.hard_invariants.workspace_leakage_count
        == policy.hard_invariants.allowlist_leakage_count
        == 0
    )
    assert policy.irrelevant_context_policy == "record_only_no_acceptance_threshold"


def test_primary_corpus_production_chunker_and_twenty_labels():
    dataset = load_live_retrieval_dataset()
    primary = tuple(d for d in dataset.documents if d.role == "primary")
    assert tuple(d.alias for d in primary) == PRIMARY_ALIASES
    chunks = prepared_live_documents(primary)
    assert [len(d.chunks) for d in chunks.sources] == [4] * 6
    assert sum(len(d.chunks) for d in chunks.sources) == 24 > RETRIEVAL_TOP_K == 5
    assert all(c.section and len(c.text.encode()) < 800 for d in chunks.sources for c in d.chunks)
    assert len([d for d in dataset.documents if d.role == "same_workspace_decoy"]) == 1
    assert len([d for d in dataset.documents if d.role == "foreign_workspace_canary"]) == 1
    assert len({q.query_id for q in dataset.queries}) == 20
    refs = {
        (d.alias, c.ordinal) for d, p in zip(primary, chunks.sources, strict=True) for c in p.chunks
    }
    for query in dataset.queries:
        assert "same_workspace_decoy" not in query.allowed_document_aliases
        assert all((r.document_alias, r.ordinal) in refs for r in query.relevant_chunks)
    filters = [q for q in dataset.queries if q.include_foreign_canary_in_allowlist]
    assert filters and all(
        r.document_alias in PRIMARY_ALIASES for q in filters for r in q.relevant_chunks
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "ordinal_99",
        "duplicate_query",
        "unknown_alias",
        "foreign_relevant",
        "decoy_relevant",
        "decoy_allowed",
        "missing_query",
        "duplicate_doc",
        "extra_foreign",
        "candidate_pool",
        "missing_canary_query",
        "unknown_field",
        "bad_type",
    ],
)
def test_retrieval_input_mutations_fail_closed(mutation):
    data = load_live_retrieval_dataset().model_dump(mode="json")
    query = data["queries"][0]
    if mutation == "ordinal_99":
        query["relevant_chunks"][0]["ordinal"] = 99
    elif mutation == "duplicate_query":
        data["queries"][1]["query_id"] = query["query_id"]
    elif mutation in ("unknown_alias", "foreign_relevant", "decoy_relevant"):
        query["relevant_chunks"][0]["document_alias"] = {
            "unknown_alias": "unknown",
            "foreign_relevant": "foreign_workspace_canary",
            "decoy_relevant": "same_workspace_decoy",
        }[mutation]
    elif mutation == "decoy_allowed":
        query["allowed_document_aliases"].append("same_workspace_decoy")
    elif mutation == "missing_query":
        data["queries"].pop()
    elif mutation == "duplicate_doc":
        data["documents"][1]["alias"] = data["documents"][0]["alias"]
    elif mutation == "extra_foreign":
        data["documents"][6]["role"] = "foreign_workspace_canary"
    elif mutation == "candidate_pool":
        data["documents"][0]["content"] += "\n# Fifth section\nExtra synthetic candidate.\n"
    elif mutation == "missing_canary_query":
        for q in data["queries"]:
            q["include_foreign_canary_in_allowlist"] = False
    elif mutation == "bad_type":
        query["relevant_chunks"][0]["ordinal"] = "0"
    else:
        data["future_option"] = True
    with pytest.raises(ValidationError):
        LiveRetrievalDatasetV1.model_validate_json(json.dumps(data), strict=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_digest", "sha256:" + "0" * 64),
        ("case_set_digest", "sha256:" + "0" * 64),
        ("embedding_profile", "wrong"),
        ("embedding_dimension", 768),
        ("normalization_version", "wrong"),
        ("chunking_version", "wrong"),
        ("retrieval_top_k", 3),
        ("expected_primary_candidate_pool_size", 25),
        ("query_count", 19),
    ],
)
def test_policy_identity_mutations_fail_closed(field, value, tmp_path):
    data = load_live_retrieval_policy().model_dump(mode="json")
    data[field] = value
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_live_retrieval_policy(path)


def test_changed_text_preserves_case_set_but_requires_new_dataset_identity(tmp_path):
    dataset = load_live_retrieval_dataset()
    data = dataset.model_dump(mode="json")
    data["documents"][0]["content"] += "\nAdditional synthetic fact within the last section.\n"
    changed = LiveRetrievalDatasetV1.model_validate_json(json.dumps(data))
    old_digests, new_digests = live_retrieval_digests(dataset), live_retrieval_digests(changed)
    assert old_digests[0] != new_digests[0] and old_digests[1] == new_digests[1]
    with pytest.raises(ValueError, match="identity mismatch"):
        load_live_retrieval_policy(dataset=changed)


def test_live_report_strict_roundtrip_and_sanitized_shape():
    payload = _report_payload()
    report = LiveRetrievalReportV1.model_validate_json(json.dumps(payload), strict=True)
    assert report.complete is report.passed is True
    assert len(report.cases) == 20
    assert '"query":' not in report.model_dump_json()
    assert (
        LiveRetrievalReportV1.model_validate_json(report.model_dump_json(), strict=True) == report
    )


def test_v2_accepted_report_and_v1_historical_readability():
    payload = _report_payload()
    historical = LiveRetrievalReportV1.model_validate_json(json.dumps(payload), strict=True)
    assert historical.exploratory is True

    payload.update(
        schema_version=2,
        evidence_scope="live_embedding_retrieval_benchmark_v2",
        exploratory=False,
        priced_attempt_count=28,
        embedding_provider_latency={
            "algorithm": "nearest-rank",
            "sample_count": 28,
            "observed_p50_ms": 0.0,
            "observed_p95_ms": 0.0,
        },
    )
    accepted_report = LiveRetrievalReportV2.model_validate_json(json.dumps(payload), strict=True)
    accepted = AcceptedLiveRetrievalBaselineV2(report=accepted_report)
    assert accepted.report.exploratory is False


def test_v2_rejects_exploratory_or_forged_invocation_summary():
    payload = _report_payload()
    payload.update(
        schema_version=2,
        evidence_scope="live_embedding_retrieval_benchmark_v2",
        priced_attempt_count=27,
        embedding_provider_latency={
            "algorithm": "nearest-rank",
            "sample_count": 28,
            "observed_p50_ms": 0.0,
            "observed_p95_ms": 0.0,
        },
    )
    with pytest.raises(ValidationError, match="priced retrieval attempts"):
        LiveRetrievalReportV2.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_digest", "sha256:" + "0" * 64),
        ("case_set_digest", "sha256:" + "0" * 64),
        ("embedding_profile", "wrong"),
        ("embedding_dimension", 768),
        ("normalization_version", "wrong"),
        ("chunking_version", "wrong"),
        ("retrieval_top_k", 3),
        ("candidate_pool_size", 23),
    ],
)
def test_live_report_identity_mutations_fail_closed(field, value):
    payload = _report_payload()
    payload[field] = value
    with pytest.raises(ValidationError):
        LiveRetrievalReportV1.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "extra", "label", "aggregate", "partial"]
)
def test_live_report_case_set_aggregate_and_partial_pass_fail_closed(mutation):
    payload = _report_payload()
    if mutation == "duplicate":
        payload["cases"][1]["query_id"] = payload["cases"][0]["query_id"]
    elif mutation == "missing":
        payload["cases"].pop()
    elif mutation == "extra":
        payload["cases"].append(payload["cases"][-1])
    elif mutation == "label":
        payload["cases"][0]["relevant_chunks"][0]["ordinal"] = 1
        payload["cases"][0]["ranked_retrieved_chunks"][0]["ordinal"] = 1
    elif mutation == "aggregate":
        payload["aggregate"]["macro_recall_at_5"] = 0.99
    else:
        payload["complete"] = False
        payload["error_category"] = "provider_failure"
        payload["passed"] = True
    with pytest.raises(ValidationError):
        LiveRetrievalReportV1.model_validate_json(json.dumps(payload), strict=True)


def _aggregate_at_thresholds(**updates):  # type: ignore[no-untyped-def]
    values = {
        "macro_recall_at_1": 0.50,
        "macro_recall_at_3": 0.75,
        "macro_recall_at_5": 0.85,
        "mean_reciprocal_rank": 0.60,
        "total_irrelevant_context_count": 999,
        "workspace_leakage_count": 0,
        "allowlist_leakage_count": 0,
    }
    values.update(updates)
    return LiveRetrievalAggregateV1(**values)


def test_live_threshold_boundaries_hard_invariants_and_irrelevant_context_policy():
    policy = load_live_retrieval_policy()
    assert live_retrieval_passes(_aggregate_at_thresholds(), policy, complete=True)
    for field in (
        "macro_recall_at_1",
        "macro_recall_at_3",
        "macro_recall_at_5",
        "mean_reciprocal_rank",
    ):
        threshold = getattr(_aggregate_at_thresholds(), field)
        assert not live_retrieval_passes(
            _aggregate_at_thresholds(**{field: threshold - 0.001}),
            policy,
            complete=True,
        )
    assert not live_retrieval_passes(
        _aggregate_at_thresholds(workspace_leakage_count=1), policy, complete=True
    )
    assert not live_retrieval_passes(
        _aggregate_at_thresholds(allowlist_leakage_count=1), policy, complete=True
    )
    assert not live_retrieval_passes(_aggregate_at_thresholds(), policy, complete=False)
