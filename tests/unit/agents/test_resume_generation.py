"""Synthetic R4.1 drafting, evidence rejection and real TeX rendering."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from app.agents.resume_generation import GenerationError, ResumeGenerationGraph
from app.domain.project_facts import MaterialRetrievalScope
from app.domain.resume_generation import (
    FixedFact,
    GenerationBudgetV1,
    GenerationInputs,
    JobInputV1,
)
from app.domain.resume_profile import ResumePreferencesV1
from app.llm.fake import ScriptedFakeChatModel
from app.llm.ports import ChatModelResult
from app.resume.template_import import parse_resume_source
from app.resume.template_render import TemplateIdentity, render_resume_tex

pytestmark = pytest.mark.asyncio
SOURCE = Path(__file__).resolve().parents[2] / "fixtures/resume/synthetic_main.tex"


def _inputs(*, kind: str = "implementation", claim: str = "Built a synthetic service"):
    source = SOURCE.read_bytes()
    preamble = source.split(b"\\begin{document}", 1)[0]
    identity = TemplateIdentity(
        source_sha256=hashlib.sha256(source).hexdigest(),
        preamble_sha256=hashlib.sha256(preamble).hexdigest(),
    )
    preview = parse_resume_source(
        source.decode(),
        expected_source_sha256=identity.source_sha256,
        expected_preamble_sha256=identity.preamble_sha256,
    )
    assert preview.content is not None
    profile = preview.content
    project = profile.projects[0].model_copy(update={"review_status": "reviewed"})
    profile = profile.model_copy(update={"projects": (project,)})
    project_id, fact_id = uuid4(), uuid4()
    inputs = GenerationInputs(
        session_id=uuid4(),
        run_id=uuid4(),
        job_text="Build a synthetic service",
        profile_content=profile,
        preferences=ResumePreferencesV1(),
        budget=GenerationBudgetV1(max_model_calls=4, max_tool_calls=0, max_cost_cny=Decimal("1")),
        facts=(FixedFact(fact_id, project_id, claim, kind, {}),),
        project_map={project.id: project_id},
        retrieval_scope=MaterialRetrievalScope(files=()),
    )
    return source, identity, inputs, project.id, fact_id


def _step(payload: dict[str, object]) -> ChatModelResult:
    return ChatModelResult(content=json.dumps(payload))


def _requirements(*, kind: str = "explicit", quote: str = "Build a synthetic service"):
    return _step(
        {
            "requirements": [{"kind": kind, "start": 0, "end": len(quote), "quote": quote}],
            "questions": [],
        }
    )


def _selection(project_id, fact_id):
    return _step(
        {
            "bullets": [
                {
                    "project_item_id": str(project_id),
                    "fact_version_id": str(fact_id),
                    "requirement_ordinals": [0],
                }
            ],
            "omitted_fact_version_ids": [],
            "questions": [],
        }
    )


async def _generate(inputs, script, *, repairs: bool = True):
    calls = []

    async def allowed():
        return True

    async def reserve():
        calls.append("repair")
        return repairs

    graph = ResumeGenerationGraph(
        model=ScriptedFakeChatModel(script), spend_allowed=allowed, reserve_repair=reserve
    )
    return await graph.generate(inputs), calls


async def test_confirmed_fact_yields_real_tex_and_human_review_coverage() -> None:
    source, identity, inputs, project_id, fact_id = _inputs()
    candidate, repairs = await _generate(inputs, [_requirements(), _selection(project_id, fact_id)])
    assert repairs == []
    assert candidate.content is not None
    assert candidate.content.projects[0].bullets[0].text == inputs.facts[0].claim
    assert candidate.coverage[0].fact_version_ids == (fact_id,)
    assert candidate.coverage[0].verification == "needs_human_review"
    assert candidate.retrieval_config_version.startswith("sha256:")
    tex = render_resume_tex(
        source_bytes=source,
        identity=identity,
        profile_content=inputs.profile_content,
        content=candidate.content,
        preferences=inputs.preferences,
    ).tex_bytes
    assert b"Built a synthetic service" in tex
    assert b"canary@example.test" in tex


@pytest.mark.parametrize(
    ("kind", "quote"),
    [("explicit", "Invented requirement"), ("inferred", "Build a synthetic service")],
)
async def test_forged_or_mislabelled_requirement_fails(kind: str, quote: str) -> None:
    _, _, inputs, _, _ = _inputs()
    with pytest.raises(GenerationError, match="invalid_job_reference"):
        await _generate(inputs, [_requirements(kind=kind, quote=quote)])


@pytest.mark.parametrize("kind", ["plan", "experiment"])
async def test_unsupported_fact_cannot_publish_content(kind: str) -> None:
    _, _, inputs, project_id, fact_id = _inputs(kind=kind)
    candidate, repairs = await _generate(
        inputs,
        [_requirements(), _selection(project_id, fact_id), _selection(project_id, fact_id)],
    )
    assert repairs == ["repair"]
    assert candidate.content is None
    assert candidate.correction_count == 1
    expected_reason = "plan_not_published" if kind == "plan" else "not_selected_for_this_job"
    assert candidate.omission_reasons[fact_id] == expected_reason
    assert "no_supported_draft_content" in candidate.questions


async def test_experiment_cannot_drop_environment_even_with_matching_metric() -> None:
    _, _, inputs, project_id, fact_id = _inputs(kind="experiment", claim="Reduced latency by 20%")
    fact = inputs.facts[0]
    inputs = GenerationInputs(
        session_id=inputs.session_id,
        run_id=inputs.run_id,
        job_text=inputs.job_text,
        profile_content=inputs.profile_content,
        preferences=inputs.preferences,
        budget=inputs.budget,
        facts=(
            FixedFact(
                fact.version_id,
                fact.project_id,
                fact.claim,
                fact.kind,
                {"environment": "synthetic staging", "metric_basis": "p95"},
            ),
        ),
        project_map=inputs.project_map,
        retrieval_scope=inputs.retrieval_scope,
    )
    candidate, _ = await _generate(
        inputs, [_requirements(), _selection(project_id, fact_id)], repairs=False
    )
    assert candidate.content is None
    assert candidate.coverage[0].verification == "unchecked"


async def test_contact_value_in_fact_fails_before_model_request() -> None:
    _, _, inputs, _, _ = _inputs(claim="Contact canary@example.test")
    with pytest.raises(Exception, match=r"protected|private|contact"):
        await _generate(inputs, [_requirements()])


async def test_invalid_selection_without_repair_keeps_only_supported_items() -> None:
    _, _, inputs, project_id, _ = _inputs()
    candidate, repairs = await _generate(
        inputs, [_requirements(), _selection(project_id, uuid4())], repairs=False
    )
    assert repairs == ["repair"]
    assert candidate.content is None
    assert candidate.coverage[0].support == "no_support_found"


async def test_jd_upload_is_bounded_and_never_truncated() -> None:
    assert JobInputV1(source="upload", filename="job.md", text="role").text == "role"
    with pytest.raises(ValueError):
        JobInputV1(source="upload", filename="job.pdf", text="role")
    with pytest.raises(ValueError):
        JobInputV1(source="paste", text="x" * (32 * 1024 + 1))


async def test_unique_exact_quote_can_ground_wrong_unicode_offsets():
    from app.agents.resume_generation import _ground_requirements
    from app.domain.resume_generation import JobRequirementV1, RequirementExtractionV1

    jd = "岗位要求:使用 Python 开发;保留原文。"
    item = JobRequirementV1(kind="explicit", start=1, end=3, quote="使用 Python 开发")
    result = _ground_requirements(jd, RequirementExtractionV1(requirements=(item,)))
    grounded = result.requirements[0]
    assert jd[grounded.start : grounded.end] == item.quote
    assert grounded.start == jd.index(item.quote)
    assert grounded.kind == "explicit"


@pytest.mark.parametrize("jd,quote", [("Python / Python", "Python"), ("Python", "Java")])
async def test_wrong_offsets_do_not_select_ambiguous_or_fabricated_quotes(jd, quote):
    from app.agents.resume_generation import _ground_requirements
    from app.domain.errors import DomainValidationError
    from app.domain.resume_generation import JobRequirementV1, RequirementExtractionV1

    item = JobRequirementV1(kind="explicit", start=1, end=2, quote=quote)
    with pytest.raises(DomainValidationError):
        _ground_requirements(jd, RequirementExtractionV1(requirements=(item,)))


@pytest.mark.parametrize("basis", ["", "  "])
async def test_empty_optional_basis_is_canonicalized_but_not_valid_inference(basis):
    from app.agents.resume_generation import _ground_requirements
    from app.domain.errors import DomainValidationError
    from app.domain.resume_generation import JobRequirementV1, RequirementExtractionV1

    item = JobRequirementV1(kind="explicit", start=0, end=1, quote="Python", inference_basis=basis)
    result = _ground_requirements("Use Python", RequirementExtractionV1(requirements=(item,)))
    assert result.requirements[0].inference_basis is None
    inferred = item.model_copy(update={"kind": "inferred"})
    with pytest.raises(DomainValidationError):
        _ground_requirements("Use Python", RequirementExtractionV1(requirements=(inferred,)))
