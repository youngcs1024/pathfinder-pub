"""Deterministic scorer, privacy, mode and publication contracts; no live calls."""

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from app.llm.fake import FakeChatModel, FakeEmbeddingModel
from tests.evals.quality_contracts import (
    QualityRetrievalMetricsV1,
    QualityRetrievalRefV1,
    QualityRetrievalRepresentationV1,
)
from tests.evals.quality_dataset import prepare_quality_mapping_sources, quality_digest
from tests.evals.quality_retrieval_support import PILOT, fake_factory, retrieval_inputs
from tests.evals.quality_run import (
    QualityRetrievalError,
    _ScopeLeak,
    run_quality_retrieval,
    score_quality_retrieval,
    write_quality_retrieval_artifact,
)


def vector(case_id, ordinals, *, prefix=None):
    inputs = retrieval_inputs(None)
    dataset, mapping = inputs["dataset"], inputs["mapping"]
    case = next(c for c in dataset.cases if c.case_id == case_id)
    prepared = prepare_quality_mapping_sources(PILOT)
    refs = []
    for ordinal in ordinals:
        content = prepared[case.resume_alias].chunks[ordinal].text
        if prefix is not None:
            content = content[:prefix]
        refs.append(
            QualityRetrievalRefV1(
                source_alias=case.resume_alias,
                chunk_ordinal=ordinal,
                exposed_digest=quality_digest(content.encode()),
                exposed_codepoints=len(content),
                exposed_bytes=len(content.encode()),
            )
        )
    return dataset, mapping, case, tuple(refs)


def test_rank_metrics_and_required_coverage_use_different_denominators():
    args = vector("mixed_alpha", [0, 3, 1])
    metrics = score_quality_retrieval(*args)
    assert metrics.recall_at_1.value == 0
    assert metrics.recall_at_3.value == metrics.recall_at_5.value == 1
    assert metrics.reciprocal_rank == 0.5
    assert metrics.required_unit_coverage.value == 1
    assert metrics.relevant_contexts == 2
    assert metrics.irrelevant_contexts == 1
    assert metrics.returned_context_bytes == sum(r.exposed_bytes for r in args[-1])


def test_multisegment_fact_requires_both_chunks_and_counts_codepoints_after_truncation():
    one = score_quality_retrieval(*vector("multi_alpha", [3]))
    both = score_quality_retrieval(*vector("multi_alpha", [3, 4]))
    cut = score_quality_retrieval(*vector("multi_alpha", [3, 4], prefix=4))
    assert one.required_unit_coverage.value == 0.5
    assert both.required_unit_coverage.value == 1
    assert cut.required_unit_coverage.value == 0
    assert cut.returned_context_bytes > 8  # Chinese code points are not UTF-8 bytes.


def test_single_unit_spanning_two_chunks_uses_union_without_filling_gaps():
    dataset, mapping, case, refs = vector("multi_alpha", [3, 4])
    first = next(u for u in dataset.units if u.unit_id == case.required_unit_ids[0])
    second = next(u for u in dataset.units if u.unit_id == case.required_unit_ids[1])
    # Include the intervening paragraph separator: neither returned chunk exposes it.
    source = prepare_quality_mapping_sources(PILOT)[case.resume_alias]
    unit = first.model_copy(
        update={
            "end": second.end,
            "quote": source.content[first.start : second.end],
        }
    )
    dataset = replace(
        dataset, units=tuple(unit if u.unit_id == unit.unit_id else u for u in dataset.units)
    )
    case = case.model_copy(update={"required_unit_ids": (unit.unit_id,)})
    assert score_quality_retrieval(dataset, mapping, case, refs).required_unit_coverage.value == 0


def test_alternative_support_only_satisfies_the_original_required_unit():
    dataset, mapping, case, refs = vector("multi_alpha", [4])
    first, second = case.required_unit_ids
    units = tuple(
        u.model_copy(update={"alternative_unit_ids": (second,)}) if u.unit_id == first else u
        for u in dataset.units
    )
    dataset = replace(dataset, units=units)
    case = case.model_copy(update={"required_unit_ids": (first,)})
    metrics = score_quality_retrieval(dataset, mapping, case, refs)
    assert metrics.required_unit_coverage.value == 1
    assert metrics.covered_unit_ids == (first,)
    foreign = next(u for u in dataset.units if u.source_alias == "resume_beta")
    dataset = replace(
        dataset,
        units=tuple(
            u.model_copy(update={"alternative_unit_ids": (foreign.unit_id,)})
            if u.unit_id == first
            else u
            for u in units
        ),
    )
    assert score_quality_retrieval(dataset, mapping, case, refs).required_unit_coverage.value == 0


@pytest.mark.parametrize(
    "case_id", ["answer_alpha", "web_alpha", "scope_missing", "scope_rejected"]
)
def test_no_relevant_evidence_is_na_and_web_never_inflates_resume_metrics(case_id):
    metrics = score_quality_retrieval(*vector(case_id, []))
    assert metrics.no_relevant_evidence
    assert metrics.recall_at_1.value is None
    assert metrics.recall_at_5.value is None
    assert metrics.reciprocal_rank is None
    if case_id == "web_alpha":
        assert metrics.web_required_units > 0
        assert metrics.required_unit_coverage.denominator == 0


def test_unjudged_is_not_irrelevant_and_ref_scope_cannot_be_overridden():
    dataset, mapping, case, refs = vector("mixed_alpha", [0])
    mapping = mapping.model_copy(
        update={
            "judgments": tuple(
                j
                for j in mapping.judgments
                if not (j.case_id == case.case_id and j.chunk_ordinal == 0)
            )
        }
    )
    metrics = score_quality_retrieval(dataset, mapping, case, refs)
    assert metrics.unjudged_contexts == 1
    assert metrics.irrelevant_contexts == 0
    with pytest.raises(_ScopeLeak):
        score_quality_retrieval(
            dataset, mapping, case, (refs[0].model_copy(update={"source_alias": "resume_beta"}),)
        )
    with pytest.raises(QualityRetrievalError):
        score_quality_retrieval(dataset, mapping, case, refs + refs)


def test_metric_contract_rejects_nonfinite_or_inconsistent_values():
    metrics = score_quality_retrieval(*vector("mixed_alpha", [1]))
    for updates in (
        {"reciprocal_rank": float("nan")},
        {"reciprocal_rank": None},
        {"no_relevant_evidence": True},
        {"covered_unit_ids": []},
        {"query": "secret_canary"},
    ):
        with pytest.raises(ValidationError):
            QualityRetrievalMetricsV1.model_validate_json(
                json.dumps(
                    {
                        **metrics.model_dump(mode="json"),
                        **updates,
                    }
                )
            )


def test_artifact_create_only_and_symlink_refusal(tmp_path):
    path = tmp_path / "result.json"
    payload = QualityRetrievalRepresentationV1(
        source_alias="resume_alpha",
        normalized_text_digest=quality_digest(b"source"),
        chunk_digests=(quality_digest(b"chunk"),),
        chunk_count=1,
    )
    write_quality_retrieval_artifact(path, payload)
    initial = path.read_bytes()
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(QualityRetrievalError, match="artifact_publication_failed"):
        write_quality_retrieval_artifact(path, payload)
    alias = tmp_path / "link.json"
    alias.symlink_to(path)
    with pytest.raises(QualityRetrievalError):
        write_quality_retrieval_artifact(alias, payload)
    assert path.read_bytes() == initial
    with pytest.raises(QualityRetrievalError):
        write_quality_retrieval_artifact(tmp_path / "raw.json", {"query": "secret_canary"})
    assert not (tmp_path / "raw.json").exists()


class NoDatabase:
    def __call__(self):
        pytest.fail("invalid configuration reached the database")


@pytest.mark.parametrize(
    "change", ["confirm", "mode", "digest", "profile", "source", "mapping", "gold"]
)
async def test_configuration_rejected_before_database_or_output_mutation(tmp_path, change):
    inputs = retrieval_inputs(tmp_path / "out")
    if change == "confirm":
        inputs["confirm_disposable_database"] = False
    elif change == "mode":
        inputs["provider_mode"] = "qwen"
    elif change == "mapping":
        inputs["mapping_rules_digest"] = quality_digest(b"wrong")
    elif change == "source":
        inputs["git_probe"] = lambda: "b" * 40
    elif change == "gold":
        inputs["dataset"] = replace(inputs["dataset"], cases=())
    else:
        key, value = (
            ("configuration_digest", quality_digest(b"wrong"))
            if change == "digest"
            else ("embedding_profile", "wrong-profile")
        )
        inputs["manifest"] = inputs["manifest"].model_copy(update={key: value})
    with pytest.raises(QualityRetrievalError, match="retrieval_preflight_failed"):
        await run_quality_retrieval(NoDatabase(), **inputs)
    assert not inputs["output_dir"].exists()


class QwenShapedChat(FakeChatModel):
    provider = "qwen"


class QwenShapedEmbedding(FakeEmbeddingModel):
    provider = "qwen"


async def test_live_requires_explicit_confirmation_even_with_provider_and_keys(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret_canary")
    factory = replace(
        fake_factory(),
        provider="qwen",
        chat_adapter=QwenShapedChat(),
        embedding_adapter=QwenShapedEmbedding(),
    )
    inputs = retrieval_inputs(tmp_path / "live", factory=factory)
    inputs["provider_mode"] = "qwen"
    with pytest.raises(QualityRetrievalError):
        await run_quality_retrieval(NoDatabase(), **inputs)
    assert not inputs["output_dir"].exists()


def test_one_contiguous_unit_split_across_chunks_requires_the_exposed_union():
    from tests.evals.test_quality_mapping import bundle

    content = "".join(chr(0x4E00 + index) for index in range(300))
    dataset, mapping, prepared = bundle(content, [(0, 300)], source_type="text")
    case = dataset.cases[0]
    refs = tuple(
        QualityRetrievalRefV1(
            source_alias="resume_alpha",
            chunk_ordinal=chunk.ordinal,
            exposed_digest=quality_digest(chunk.text.encode()),
            exposed_codepoints=len(chunk.text),
            exposed_bytes=len(chunk.text.encode()),
        )
        for chunk in prepared["resume_alpha"].chunks
    )
    assert len(refs) > 1
    assert (
        score_quality_retrieval(dataset, mapping, case, refs[:1]).required_unit_coverage.value == 0
    )
    assert score_quality_retrieval(dataset, mapping, case, refs).required_unit_coverage.value == 1
