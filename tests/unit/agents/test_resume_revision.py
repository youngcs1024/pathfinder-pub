"""Synthetic scoped revisions, attested numbers and bounded model correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.agents.resume_revision import ResumeRevisionGraph
from app.domain.errors import DomainValidationError
from app.domain.resume_profile import ResumePreferencesV1
from app.domain.resume_revision import (
    ContentFeedbackV1,
    PatchV1,
    PreferenceFeedbackV1,
    RevisionCandidateV1,
    RevisionInputs,
    UserFactInputV1,
    apply_scoped_patches,
    user_fact_issues,
)
from app.domain.run_payloads import ResumeRevisionCandidateOutputV1
from app.domain.runs import RunStatus
from app.llm.fake import ScriptedFakeChatModel
from app.llm.ports import ChatModelResult
from app.resume.template_import import parse_resume_source
from app.worker.contracts import RunExecutionResult

SOURCE = Path(__file__).resolve().parents[2] / "fixtures/resume/synthetic_main.tex"


def _content():
    source = SOURCE.read_bytes()
    return parse_resume_source(
        source.decode(),
        expected_source_sha256=hashlib.sha256(source).hexdigest(),
        expected_preamble_sha256=hashlib.sha256(
            source.split(b"\\begin{document}", 1)[0]
        ).hexdigest(),
    ).content


def _patch(content, fact_id, *, text="Built 12 synthetic jobs"):
    return PatchV1(
        operation="replace_text",
        item_id=content.projects[0].bullet_ids[0],
        field="bullet",
        text=text,
        fact_version_ids=(fact_id,),
    )


def test_scoped_patch_preserves_personal_and_other_items() -> None:
    content = _content()
    assert content is not None
    fact_id = uuid4()
    target = content.projects[0].bullet_ids[0]
    revised, diff = apply_scoped_patches(
        content,
        (_patch(content, fact_id),),
        (target,),
        ResumePreferencesV1(),
        frozenset({fact_id}),
        {fact_id: ("Built 12 synthetic jobs", "implementation")},
    )
    assert revised.projects[0].bullets[0].text == "Built 12 synthetic jobs"
    assert revised.display_name == content.display_name
    assert revised.contact == content.contact
    assert revised.projects[0].bullet_ids == content.projects[0].bullet_ids
    assert diff.changes[0]["item_id"] == str(target)


def test_scope_lock_and_unattested_number_block_publication() -> None:
    content = _content()
    assert content is not None
    fact_id = uuid4()
    target = content.projects[0].bullet_ids[0]
    args = (content, (_patch(content, fact_id),), (target,))
    with pytest.raises(DomainValidationError, match="lock"):
        apply_scoped_patches(
            *args,
            ResumePreferencesV1(locked_item_ids=(target,)),
            frozenset({fact_id}),
            {fact_id: ("Built 12 synthetic jobs", "implementation")},
        )
    with pytest.raises(DomainValidationError, match="numbers"):
        apply_scoped_patches(
            content,
            (_patch(content, fact_id, text="Built 99 synthetic jobs"),),
            (target,),
            ResumePreferencesV1(),
            frozenset({fact_id}),
            {fact_id: ("Built 12 synthetic jobs", "implementation")},
        )
    with pytest.raises(DomainValidationError, match="scope"):
        apply_scoped_patches(
            content,
            (_patch(content, fact_id),),
            (content.projects[0].id,),
            ResumePreferencesV1(),
            frozenset({fact_id}),
            {fact_id: ("Built 12 synthetic jobs", "implementation")},
        )


def test_unverified_metric_requires_environment_scope_and_basis() -> None:
    fact = UserFactInputV1(
        project_id=uuid4(),
        scope="session",
        claim="Reduced latency by 12%",
        kind="implementation",
    )
    assert user_fact_issues(fact) == (
        "metric_environment_missing",
        "metric_scope_missing",
        "metric_basis_missing",
    )
    assert not user_fact_issues(
        fact.model_copy(
            update={
                "environment": "synthetic benchmark",
                "fact_scope": "API requests",
                "metric_basis": "p95 before and after",
            }
        )
    )


@pytest.mark.asyncio
async def test_model_patch_repairs_once_then_keeps_original_on_invalid_scope() -> None:
    content = _content()
    assert content is not None
    fact_id, session_id, run_id, feedback_id = (uuid4() for _ in range(4))
    target = content.projects[0].bullet_ids[0]
    request = ContentFeedbackV1(
        expected_session_revision=1,
        base_version_id=uuid4(),
        target_item_ids=(target,),
        instruction="Make this bullet clearer",
    )
    inputs = RevisionInputs(
        session_id=session_id,
        run_id=run_id,
        feedback_id=feedback_id,
        base_version_id=request.base_version_id,
        base_content=content,
        profile_content=content,
        preferences=ResumePreferencesV1(),
        target_item_ids=(target,),
        request=request,
        permitted_fact_ids=frozenset({fact_id}),
        fact_claims=((fact_id, "Built 12 synthetic jobs", "implementation", {}),),
        instruction=request.instruction,
    )
    bad = {
        "patches": [
            _patch(content, fact_id, text="Built 99 synthetic jobs").model_dump(mode="json")
        ]
    }
    repairs = []

    async def allowed():
        return True

    async def reserve():
        repairs.append(1)
        return True

    graph = ResumeRevisionGraph(
        model=ScriptedFakeChatModel(
            [
                ChatModelResult(content=json.dumps(bad)),
                ChatModelResult(content=json.dumps(bad)),
            ]
        ),
        spend_allowed=allowed,
        reserve_repair=reserve,
    )
    candidate = await graph.generate(inputs)
    assert candidate.content is None
    assert candidate.correction_count == 1
    assert len(repairs) == 1
    assert candidate.questions == ("revision_conflicts_with_scope_facts_or_locks",)


def test_preference_scope_selects_versioned_override_or_global_profile() -> None:
    import json

    for scope, expected in (
        ("round", "JobPreferenceOverrideV1"),
        ("session", "JobPreferenceOverrideV1"),
        ("global", "ResumePreferencesV1"),
    ):
        data = {
            "kind": "preference",
            "expected_session_revision": 2,
            "base_version_id": str(uuid4()),
            "scope": scope,
            "preferences": {"page_target": 2},
        }
        if scope == "global":
            data["expected_global_preference_version"] = 1
        request = PreferenceFeedbackV1.model_validate_json(json.dumps(data))
        assert type(request.preferences).__name__ == expected


@pytest.mark.asyncio
async def test_non_target_draft_lock_does_not_block_scoped_edit() -> None:
    profile = _content()
    assert profile is not None
    project = profile.projects[0]
    generated_bullet_id = uuid4()
    current_project = project.model_copy(
        update={"bullet_ids": (project.bullet_ids[0], generated_bullet_id)}
    )
    current = profile.model_copy(update={"projects": (current_project,)})
    fact_id = uuid4()
    request = ContentFeedbackV1(
        expected_session_revision=2,
        base_version_id=uuid4(),
        target_item_ids=(project.bullet_ids[0],),
        patches=(_patch(current, fact_id),),
    )
    inputs = RevisionInputs(
        session_id=uuid4(),
        run_id=uuid4(),
        feedback_id=uuid4(),
        base_version_id=request.base_version_id,
        base_content=current,
        profile_content=profile,
        preferences=ResumePreferencesV1(locked_item_ids=(generated_bullet_id,)),
        target_item_ids=request.target_item_ids,
        request=request,
        permitted_fact_ids=frozenset({fact_id}),
        fact_claims=((fact_id, "Built 12 synthetic jobs", "implementation", {}),),
        instruction=None,
    )

    async def allowed():
        return True

    graph = ResumeRevisionGraph(
        model=ScriptedFakeChatModel([ChatModelResult(content='{"patches":[]}')]),
        spend_allowed=allowed,
        reserve_repair=allowed,
    )
    candidate = await graph.generate(inputs)
    assert candidate.content is not None
    assert candidate.content.projects[0].bullet_ids[1] == generated_bullet_id
    assert candidate.content.projects[0].bullets[1] == current.projects[0].bullets[1]


def test_revision_candidate_satisfies_strict_worker_output_contract() -> None:
    candidate = RevisionCandidateV1(
        content=None,
        patches=(),
        diff=(),
        impact=(),
        questions=("Need a clear target",),
        correction_count=0,
        prompt_version="synthetic",
        model_id="fake",
        retrieval_config_version="synthetic",
    )
    output = ResumeRevisionCandidateOutputV1(payload=candidate)
    assert RunExecutionResult(status=RunStatus.COMPLETED, result=output).result == output
