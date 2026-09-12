"""Hand-calculated E4.8 oracles; fake metadata, never human/live evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal

import pytest

from tests.evals.quality_contracts import (
    HumanAnnotationV1,
    QualityGenerationReportV1,
    QualityRetrievalReportV1,
)
from tests.evals.quality_dataset import load_quality_dataset, quality_digest
from tests.evals.quality_score import (
    QualityScoreError,
    samples,
    score_quality,
    write_quality_score,
)
from tests.evals.test_quality_contracts import (
    DIGEST,
    FIXTURE,
    SHA,
    annotation_payload,
    observation_payload,
    run_payload,
)


def parse(model, payload):
    return model.model_validate_json(json.dumps(payload))


def inputs(statuses=("succeeded",), *, annotations=True, experiment="example_run"):
    dataset = load_quality_dataset(FIXTURE)
    manifest = run_payload()
    manifest.update(
        experiment_id=experiment,
        repeat_count=len(statuses),
        execution_order=[
            {"case_id": "claim_strength", "repeat_index": i} for i in range(len(statuses))
        ],
    )
    cases, reviews, files = [], [], []
    for i, status in enumerate(statuses):
        observation = observation_payload()
        observation.update(experiment_id=experiment, repeat_index=i, status=status)
        observation["stage_times"][0]["seconds"] = float(i + 1)
        if status != "succeeded":
            observation.update(output_digest=None, assessment_required=False)
            observation["failure_type"] = "provider" if status == "failed" else None
        if status == "not_run":
            observation["stage_times"] = []
        if observation["output_digest"]:
            files.append({"name": f"output-{i:04d}.json", "digest": DIGEST, "byte_count": 1})
            if annotations:
                review = annotation_payload()
                review.update(experiment_id=experiment, repeat_index=i)
                reviews.append(parse(HumanAnnotationV1, review))
        cases.append({"observation": observation})
    usage = {
        "provider_attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "unknown_usage_attempts": 0,
        "cost": observation_payload()["cost"],
        "accounting_complete": True,
    }
    payload = {
        "start": {
            "manifest": manifest,
            "policy": {"retrieval": {"unknown_attempt_reserve_cny": "0.1"}},
            "mapping_digest": DIGEST,
            "plan_prompt_digest": DIGEST,
            "research_prompt_digest": DIGEST,
            "writer_prompt_digest": DIGEST,
        },
        "representations": [],
        "representation_complete": True,
        "ingestion_usage": usage,
        "total_usage": usage,
        "ingestion_seconds": 0.0,
        "cases": cases,
        "private_files": files,
        "executed": sum(s != "not_run" for s in statuses),
        "failed": statuses.count("failed"),
        "not_run": statuses.count("not_run"),
        "evidence_valid": True,
        "measurement_complete": "not_run" not in statuses,
        "stop_reason": None,
    }
    return dataset, parse(QualityGenerationReportV1, payload), tuple(reviews)


def changed_review(review, **changes):
    payload = review.model_dump(mode="json")
    payload.update(changes)
    return parse(HumanAnnotationV1, payload)


def score(data):
    dataset, report, reviews = data
    return score_quality(dataset, report, reviews, scorer_source_sha=SHA)


def test_hand_calculated_counts_keep_failure_and_not_run():
    dataset, report, reviews = inputs(("succeeded", "succeeded", "failed", "not_run"))
    reviews = (
        reviews[0],
        changed_review(
            reviews[1],
            facts=[{"fact_id": "claim_one", "judgment": "unsupported"}],
            citations=[
                {"fact_id": "claim_one", "citation_id": "citation_one", "judgment": "contradicted"}
            ],
            covered_unit_ids=["course_pg"],
            missing_unit_ids=["role_pg"],
            business_result="failure",
        ),
    )
    result = score((dataset, report, reviews))
    s = result.summary
    assert (s.coverage.planned, s.coverage.executed, s.coverage.failed, s.coverage.not_run) == (
        4,
        3,
        1,
        1,
    )
    assert (s.business_success.numerator, s.business_success.denominator) == (1, 3)
    assert s.fact_support.value == s.citation_support.value == 0.5
    assert (
        s.required_evidence.numerator,
        s.required_evidence.denominator,
        s.required_evidence.unassessed_count,
    ) == (3, 8, 4)
    assert (
        s.stage_times["total"].sample_count,
        s.stage_times["total"].p50,
        s.stage_times["total"].p95,
    ) == (3, 2, 3)
    assert s.independent_family_count == 1
    assert result.measurement_complete is False
    assert result.semantic_quality_claim is False
    assert result.meets_quality_target is None


def test_all_failures_are_business_failures_with_no_fact_denominator():
    result = score(inputs(("failed", "failed")))
    assert result.summary.business_success.value == 0
    assert result.summary.business_success.denominator == 2
    assert result.summary.fact_support.value is None
    assert result.summary.fact_support.not_applicable_reason == "zero_denominator"
    assert result.summary.cost.per_success_known_cny is None
    assert result.measurement_complete  # Failure is a completed observation, not a quality pass.


def test_empty_annotations_do_not_invent_fact_inventory_or_success():
    result = score(inputs(annotations=False))
    assert result.summary.business_success.unassessed_count == 1
    assert result.summary.business_success.numerator == 0
    assert result.summary.facts.unassessed == 0  # Unknown number of facts is not fabricated.
    assert result.summary.unknown_fact_inventory_outputs == 1
    assert result.summary.coverage.unassessed == 1
    assert not result.measurement_complete


def test_not_assessable_is_not_unreviewed_or_supported():
    dataset, report, reviews = inputs()
    review = changed_review(
        reviews[0],
        facts=[{"fact_id": "claim_one", "judgment": "not_assessable"}],
        business_result="not_assessable",
    )
    result = score((dataset, report, (review,)))
    assert result.summary.facts.not_assessable == 1
    assert result.summary.facts.unassessed == 0
    assert result.summary.fact_support.value is None
    assert result.summary.coverage.assessed == 1
    assert result.summary.business_success.unassessed_count == 0
    assert result.summary.business_not_assessable == 1


def test_partial_review_counts_known_judgments_and_unreviewed_units():
    dataset, report, reviews = inputs()
    review = changed_review(
        reviews[0],
        covered_unit_ids=["course_pg"],
        unassessed_fact_count=2,
        assessment_complete=False,
    )
    result = score((dataset, report, (review,)))
    assert result.summary.fact_support.unassessed_count == 2
    assert result.summary.required_evidence.unassessed_count == 1
    assert not result.measurement_complete
    with pytest.raises(QualityScoreError, match="incomplete_required_evidence"):
        score((dataset, report, (changed_review(reviews[0], covered_unit_ids=[]),)))


def test_zero_required_units_have_null_coverage():
    dataset, report, reviews = inputs()
    dataset = replace(
        dataset, cases=tuple(c.model_copy(update={"required_unit_ids": ()}) for c in dataset.cases)
    )
    result = score((dataset, report, reviews))
    assert result.summary.required_evidence.value is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment_id", "wrong"),
        ("case_id", "wrong"),
        ("repeat_index", 2),
        ("output_digest", "sha256:" + "b" * 64),
        ("rubric_version", "wrong"),
        ("covered_unit_ids", ["foreign_unit"]),
    ],
)
def test_annotation_identity_or_scope_is_rejected(field, value):
    dataset, report, reviews = inputs()
    with pytest.raises(QualityScoreError, match="invalid_scoring_input"):
        score((dataset, report, (changed_review(reviews[0], **{field: value}),)))


def test_duplicate_and_no_output_reviews_rejected():
    dataset, report, reviews = inputs()
    with pytest.raises(QualityScoreError, match="invalid_scoring_input"):
        score((dataset, report, reviews + reviews))
    _, failed, _ = inputs(("failed",))
    with pytest.raises(QualityScoreError, match="invalid_scoring_input"):
        score((dataset, failed, reviews))


def test_conflicting_reviewers_are_not_majority_voted():
    dataset, report, reviews = inputs()
    other = changed_review(reviews[0], reviewer_id="second", business_result="failure")
    with pytest.raises(QualityScoreError, match="conflicting_annotations"):
        score((dataset, report, (*reviews, other)))
    identical = changed_review(reviews[0], reviewer_id="second")
    assert score((dataset, report, (*reviews, identical))).summary.facts.supported == 1


def test_independent_review_policy_requires_second_review():
    dataset, report, reviews = inputs()
    dataset = replace(
        dataset,
        rubric=dataset.rubric.model_copy(
            update={"review_policy": "independent_review_with_adjudication"}
        ),
    )
    result = score((dataset, report, reviews))
    assert result.summary.coverage.assessed == 0
    assert result.summary.fact_support.value is None


def test_multi_tag_overlap_does_not_duplicate_overall_counts():
    dataset, report, reviews = inputs(("succeeded", "succeeded"))
    dataset = replace(
        dataset,
        cases=tuple(c.model_copy(update={"tags": ("tag_a", "tag_b")}) for c in dataset.cases),
    )
    result = score((dataset, report, reviews))
    tags = [g for g in result.groups if g.dimension == "tag"]
    assert len(tags) == 2 and all(g.overlapping for g in tags)
    assert sum(g.summary.coverage.planned for g in tags) == 4
    assert result.summary.coverage.planned == 2
    assert len([g for g in result.groups if g.dimension == "family"]) == 1


@pytest.mark.parametrize(
    "behavior,label,metric",
    [
        ("state_answer_not_in_material", "appropriate", "no_answer_handling"),
        ("preserve_responsibility_strength", "unnecessary_refusal", "unnecessary_refusal"),
    ],
)
def test_answer_policy_uses_explicit_case_expectation(behavior, label, metric):
    dataset, report, reviews = inputs()
    dataset = replace(
        dataset,
        cases=tuple(c.model_copy(update={"expected_behavior": behavior}) for c in dataset.cases),
    )
    result = score((dataset, report, (changed_review(reviews[0], insufficiency=label),)))
    assert getattr(result.summary, metric).value == 1
    assert result.summary.unmapped_answer_cases == 0


def test_unmapped_behavior_does_not_guess_answerability():
    dataset, report, reviews = inputs()
    dataset = replace(
        dataset,
        cases=tuple(
            c.model_copy(update={"expected_behavior": "unknown_behavior"}) for c in dataset.cases
        ),
    )
    result = score((dataset, report, reviews))
    assert result.summary.unmapped_answer_cases == 1
    assert result.summary.no_answer_handling.value is None
    assert result.summary.unnecessary_refusal.value is None


def test_unknown_cost_and_ingestion_scope_are_preserved():
    dataset, report, reviews = inputs()
    payload = report.model_dump(mode="json")
    cost = {
        "scope": "all",
        "known_cost_cny": "0.2",
        "priced_attempts": 1,
        "unknown_cost_attempts": 1,
        "total_cost_cny": None,
    }
    payload["cases"][0]["observation"].update(provider_attempts=2, model_calls=1, cost=cost)
    payload["total_usage"].update(provider_attempts=2, cost=cost)
    result = score((dataset, parse(QualityGenerationReportV1, payload), reviews))
    assert result.total_cost.total_cost_cny is None
    assert result.summary.cost.per_success_known_cny == Decimal("0.2")
    assert result.summary.cost.component_costs is None
    assert result.ingestion_cost.known_cost_cny == 0
    assert result.summary.provider_attempts == 2


def test_nearest_rank_hand_calculation_and_empty_samples():
    s = samples(range(1, 21))
    assert (s.mean, s.p50, s.p95) == (10.5, 10, 19)
    assert samples([]).null_reason == "no_samples"


def test_retrieval_macro_excludes_na_but_micro_keeps_counts():
    dataset, generation, _ = inputs(("succeeded", "succeeded"), annotations=False)
    payload = generation.model_dump(mode="json")
    cases = []
    for i, original in enumerate(payload["cases"]):
        observation = original["observation"]
        observation.update(output_digest=None, assessment_required=False)

        def r(n, d):
            return {
                "numerator": n,
                "denominator": d,
                "value": n / d if d else None,
                "unassessed_count": 0,
                "not_applicable_reason": None if d else "zero_denominator",
            }

        metrics = {
            "recall_at_1": r(0, 2 if i == 0 else 0),
            "recall_at_3": r(0, 2 if i == 0 else 0),
            "recall_at_5": r(0, 2 if i == 0 else 0),
            "reciprocal_rank": 0.0 if i == 0 else None,
            "required_unit_coverage": r(0, 1),
            "covered_unit_ids": [],
            "web_required_units": 1,
            "relevant_contexts": 0,
            "irrelevant_contexts": 0,
            "unjudged_contexts": 0,
            "returned_context_bytes": 0,
            "no_relevant_evidence": i == 1,
        }
        cases.append({"observation": observation, "refs": [], "metrics": metrics})
    retrieval = {
        k: payload[k]
        for k in (
            "representations",
            "representation_complete",
            "ingestion_usage",
            "total_usage",
            "ingestion_seconds",
            "executed",
            "failed",
            "not_run",
            "evidence_valid",
            "measurement_complete",
            "stop_reason",
        )
    }
    retrieval.update(
        manifest=payload["start"]["manifest"],
        mapping_digest=DIGEST,
        policy={"unknown_attempt_reserve_cny": "0.1"},
        semantic_quality_claim=False,
        cases=cases,
        scope_leakage_count=0,
        full_coverage_cases=r(0, 2),
    )
    result = score((dataset, parse(QualityRetrievalReportV1, retrieval), ()))
    assert result.summary.retrieval_macro["recall_at_1"].sample_count == 1
    assert result.summary.retrieval_macro["recall_at_1"].missing_count == 1
    assert result.summary.retrieval_micro["recall_at_1"].denominator == 2

    for i, case in enumerate(retrieval["cases"]):
        denominator = 1 if i == 0 else 4
        case["refs"] = [
            {
                "source_alias": "resume_alpha",
                "chunk_ordinal": 0,
                "exposed_digest": DIGEST,
                "exposed_codepoints": 1,
                "exposed_bytes": 1,
            }
        ]
        case["metrics"].update(
            recall_at_1=r(1, denominator),
            recall_at_3=r(1, denominator),
            recall_at_5=r(1, denominator),
            reciprocal_rank=1.0,
            no_relevant_evidence=False,
            relevant_contexts=1,
            returned_context_bytes=1,
        )
    mixed = score((dataset, parse(QualityRetrievalReportV1, retrieval), ()))
    assert mixed.summary.retrieval_macro["recall_at_1"].mean == 0.625
    assert mixed.summary.retrieval_micro["recall_at_1"].value == 0.4


def test_publication_create_only_safe_and_no_body(tmp_path, monkeypatch):
    result = score(inputs())
    path = tmp_path / "score.json"
    digest = write_quality_score(path, result)
    assert digest == quality_digest(path.read_bytes())
    assert path.stat().st_mode & 0o777 == 0o600
    for prohibited in ("query", "facts_text", "reviewer_id", "reason_codes"):
        assert f'"{prohibited}"' not in path.read_text()
    with pytest.raises(QualityScoreError, match="score_publication_failed"):
        write_quality_score(path, result)
    monkeypatch.setenv("PF_TEST_SECRET", "example_run")
    with pytest.raises(QualityScoreError, match=r"^score_publication_failed$"):
        write_quality_score(tmp_path / "leak.json", result)
    assert not (tmp_path / "leak.json").exists()


def test_symlink_and_wrong_public_contract_rejected(tmp_path):
    result = score(inputs())
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(QualityScoreError, match="score_publication_failed"):
        write_quality_score(tmp_path / "link" / "score.json", result)
    with pytest.raises(QualityScoreError, match="score_publication_failed"):
        write_quality_score(tmp_path / "body.json", {"query": "private body"})


def test_changed_scorer_requires_reason_and_preserves_identity():
    dataset, report, reviews = inputs()
    with pytest.raises(QualityScoreError, match="invalid_scoring_input"):
        score_quality(dataset, report, reviews, scorer_source_sha="b" * 40)
    result = score_quality(
        dataset, report, reviews, scorer_source_sha="b" * 40, scorer_change_reason="new_scorer"
    )
    assert result.execution_source_sha == SHA
    assert result.scorer_source_sha == "b" * 40


def oracle_check(grader, cases):
    # This fixed oracle is deliberately independent of production scorer calculations.
    if set(cases) != {"supported", "unsupported", "contradicted"}:
        raise AssertionError("missing_negative_control")
    for label, expected in (("supported", 1.0), ("unsupported", 0.0), ("contradicted", 0.0)):
        assert grader(cases[label]).summary.fact_support.value == expected


def test_grader_self_check_detects_constant_true_and_removed_negative():
    dataset, report, reviews = inputs()
    cases = {
        label: (
            dataset,
            report,
            (changed_review(reviews[0], facts=[{"fact_id": "claim_one", "judgment": label}]),),
        )
        for label in ("supported", "unsupported", "contradicted")
    }
    oracle_check(score, cases)
    with pytest.raises(AssertionError):
        oracle_check(lambda _: score(cases["supported"]), cases)
    with pytest.raises(AssertionError, match="missing_negative_control"):
        oracle_check(score, {k: v for k, v in cases.items() if k != "unsupported"})


def test_all_not_run_has_no_cost_or_latency_denominator():
    result = score(inputs(("not_run", "not_run")))
    assert result.summary.business_success.value is None
    assert result.summary.cost.per_executed_known_cny is None
    assert result.summary.stage_times["total"].sample_count == 0
    assert result.summary.coverage.not_run == 2
    assert not result.measurement_complete


def test_nonzero_ingestion_is_not_allocated_to_task_average():
    dataset, report, reviews = inputs()
    payload = report.model_dump(mode="json")
    cost = {
        "scope": "all",
        "known_cost_cny": "0.3",
        "priced_attempts": 1,
        "unknown_cost_attempts": 0,
        "total_cost_cny": "0.3",
    }
    payload["ingestion_usage"].update(cost=cost, provider_attempts=1)
    payload["total_usage"].update(cost=cost, provider_attempts=1)
    result = score((dataset, parse(QualityGenerationReportV1, payload), reviews))
    assert (
        result.ingestion_cost.known_cost_cny == result.total_cost.known_cost_cny == Decimal("0.3")
    )
    assert result.summary.cost.per_executed_known_cny == 0


def test_cancellation_is_not_converted_to_scoring_error(monkeypatch):
    import asyncio

    def cancelled(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr("tests.evals.quality_score.validate_quality_annotations", cancelled)
    with pytest.raises(asyncio.CancelledError):
        score(inputs())


def test_annotation_jsonl_import_empty_malformed_duplicate_and_size(tmp_path):
    from tests.evals.quality_score import load_quality_annotations

    path = tmp_path / "labels.jsonl"
    path.write_text("\n")
    assert load_quality_annotations(path) == ()
    review = inputs()[2][0]
    path.write_text(review.model_dump_json() + "\n")
    assert load_quality_annotations(path) == (review,)
    for content in ("private_body_secret", (review.model_dump_json() + "\n") * 2, "x" * 1_000_001):
        path.write_text(content)
        with pytest.raises(QualityScoreError, match=r"^invalid_annotation_file$") as error:
            load_quality_annotations(path)
        assert "private_body_secret" not in str(error.value)
        assert str(path) not in str(error.value)
