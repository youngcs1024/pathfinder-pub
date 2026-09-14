"""E7-A.1 semantics and legacy characterization, not an assessment implementation.

Expected labels are reviewed test data. Runtime inputs below use only requests
and source bodies; no model, database, private artifact or new graph is needed.
"""

import copy
import json
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import pytest
from pydantic import Field, StringConstraints, ValidationError, model_validator

from app.agents.research_contracts import (
    ApplicationDraftV1,
    EvidenceValidationNodeInputV1,
    ResearchCitationV1,
    ResearchClaimV1,
    ResearchEvidenceV1,
    ResearchEvidenceV2,
    ResearchLimitationV1,
    ResearchPlanV1,
    ResearchRequestV1,
    ResearchSourceV1,
    WriteReportNodeInputV1,
    WriteReportNodeOutputV1,
)
from app.agents.research_nodes import (
    DeterministicEvidenceValidationNode,
    ResearchWriterNodeError,
    _validate_writer_grounding,
    _writer_user_message,
)
from tests.evals.contracts import EvalContractModel, EvalIdentifier
from tests.evals.quality_dataset import (
    load_quality_dataset,
    quality_digest,
    quality_identity_digest,
)
from tests.evals.quality_experiment_implementation import (
    load_implementation_package,
    validate_implementation_package,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evals/datasets/quality_expanded_v1"
ARTIFACT = ROOT / "evals/experiments/e7a-semantics-v1.json"
BASELINE = ROOT / "evals/baselines/quality/e410-agent-v1.json"
PACKAGE = ROOT / "evals/experiments/e65-evidence-sufficiency-implementation-v1.json"
SEMANTICS_DIGEST = "sha256:89dda5e12a1bc6f0ad8ff028e0fd20421e3636150a6bb989b601d51e1d708bcd"
Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
Codes = Annotated[tuple[EvalIdentifier, ...], Field(min_length=1, max_length=16)]

# Independent, reviewed expectations. These are never a runtime classifier.
EXPECTED = {
    "e7a_unrelated_only": ("unrelated_only", "insufficient", False),
    "start_unknown": ("partial_support", "partial", False),
    "pager_shadow": ("claim_strength", "sufficient", True),
    "office_conflict": ("material_conflict", "conflicting", False),
    "e7a_resume_only": ("legitimate_resume_only", "sufficient", True),
    "sharding_gap": ("known_skill_gap", "sufficient", True),
}
OBSERVED = {"pager_shadow", "office_conflict", "sharding_gap"}
SYNTHETIC = {
    "e7a_unrelated_only": {
        "query": "请根据资料为数据库运维岗位写一份有依据的申请草稿。",
        "alias": "e7a_unrelated_resume",
        "body": "课程记录\uff1a只做过水彩配色练习。",
    },
    "e7a_resume_only": {
        "query": "仅根据简历写一份通用申请草稿\uff0c不描述任何具体岗位的匹配程度。",
        "alias": "e7a_resume_only_resume",
        "body": "课程项目中编写了 CSV 导入脚本\uff0c并用三份合成文件检查重复行。",
    },
}


class SourceReference(EvalContractModel):
    alias: EvalIdentifier
    kind: Literal["resume", "web"]
    digest: Digest


class SemanticCase(EvalContractModel):
    case_id: EvalIdentifier
    category: EvalIdentifier
    origin: Literal["e4_observed", "dev_code_characterization", "synthetic_regression"]
    source_case_id: EvalIdentifier | None
    sources: tuple[SourceReference, ...] = Field(min_length=1, max_length=2)
    old_evidence_sufficient: Literal[True]
    old_behavior: Codes
    observed_output_digest: Digest | None
    observed_review_digest: Digest | None
    expected_outcome: Literal["sufficient", "partial", "insufficient", "conflicting"]
    report_policy: Codes
    draft_eligible: bool
    followup_policy: Literal["none", "retrievable_gap_once_within_shared_budget"]
    forbidden_inferences: Codes

    @model_validator(mode="after")
    def consistent_policy(self):
        if self.draft_eligible and self.expected_outcome != "sufficient":
            raise ValueError("ineligible_outcome")
        if self.expected_outcome in {"sufficient", "conflicting"}:
            if self.followup_policy != "none":
                raise ValueError("invalid_followup")
        observed = self.origin == "e4_observed"
        if observed != (self.observed_output_digest is not None):
            raise ValueError("invalid_observation_identity")
        if observed != (self.observed_review_digest is not None):
            raise ValueError("invalid_review_identity")
        if len({source.alias for source in self.sources}) != len(self.sources):
            raise ValueError("duplicate_source")
        for codes in (self.old_behavior, self.report_policy, self.forbidden_inferences):
            if len(set(codes)) != len(codes):
                raise ValueError("duplicate_policy")
        return self


class Semantics(EvalContractModel):
    artifact_kind: Literal["e7a_semantics_v1"]
    source_commit: Literal["b2337ef60793223852ed01b81b7baa218c59ae97"]
    implementation_package_digest: Digest
    dataset_manifest_digest: Digest
    dataset_cases_digest: Digest
    accepted_baseline_digest: Digest
    policy: Codes
    cases: tuple[SemanticCase, ...] = Field(min_length=6, max_length=6)
    verification_limit: Literal["semantics_and_legacy_characterization_only"]
    candidate_execution: Literal["NOT_RUN"]
    adoption: Literal["NOT_RUN"]
    deployment: Literal["NOT_RUN"]


def checked(raw):
    """Test-local artifact audit; no runtime policy or externally exposed loader."""
    artifact = Semantics.model_validate_json(json.dumps(raw))
    assert tuple(case.case_id for case in artifact.cases) == tuple(EXPECTED)
    assert artifact.implementation_package_digest == validate_implementation_package(
        load_implementation_package(PACKAGE)
    )
    assert artifact.dataset_manifest_digest == quality_digest(
        (DATASET / "manifest.json").read_bytes()
    )
    assert artifact.dataset_cases_digest == quality_digest((DATASET / "cases.jsonl").read_bytes())
    assert artifact.accepted_baseline_digest == quality_digest(BASELINE.read_bytes())
    dataset = load_quality_dataset(DATASET)
    records = {case.case_id: case for case in dataset.cases}
    sources = {source.alias: source for source in dataset.manifest.sources}
    slots = {
        slot["case_id"]: slot
        for slot in json.loads(BASELINE.read_bytes())["candidate"]["review"]["slots"]
    }
    for case in artifact.cases:
        assert (case.category, case.expected_outcome, case.draft_eligible) == EXPECTED[case.case_id]
        if case.case_id in SYNTHETIC:
            fixture = SYNTHETIC[case.case_id]
            assert case.origin == "synthetic_regression" and case.source_case_id is None
            assert case.case_id not in records and case.case_id not in slots
            assert len(case.sources) == 1
            source = case.sources[0]
            assert source.alias == fixture["alias"] and source.kind == "resume"
            assert source.digest == quality_digest(fixture["body"].encode())
        else:
            assert case.source_case_id == case.case_id
            record = records[case.source_case_id]
            assert record.split == "dev" and record.scope_expectation == "allowed"
            aliases = tuple(a for a in (record.resume_alias, record.web_scenario_alias) if a)
            assert tuple(source.alias for source in case.sources) == aliases
            for source in case.sources:
                registered = sources[source.alias]
                assert source.kind == registered.kind
                assert source.digest == quality_digest((DATASET / registered.path).read_bytes())
            if case.case_id in OBSERVED:
                assert case.origin == "e4_observed"
                assert case.observed_output_digest == slots[case.case_id]["output_digest"]
                assert case.observed_review_digest == slots[case.case_id]["recheck_digest"]
                label = (
                    "observed_conflict_preserved"
                    if case.case_id == "office_conflict"
                    else "observed_overclaim"
                )
                assert label in case.old_behavior
            else:
                assert case.origin == "dev_code_characterization"
                assert case.case_id not in slots
    return artifact


def runtime_input(case_id, records=None):
    """Project raw task/source data only; never load the semantics artifact or gold units."""
    if case_id in SYNTHETIC:
        fixture = SYNTHETIC[case_id]
        query, application = fixture["query"], True
        bodies = [(fixture["alias"], "resume", fixture["body"])]
    else:
        if records is None:
            records = {
                row["case_id"]: row
                for row in map(json.loads, (DATASET / "cases.jsonl").read_text().splitlines())
            }
        record = records[case_id]
        query, application = record["query"], record["mode"] == "application"
        manifest = json.loads((DATASET / "manifest.json").read_bytes())
        sources = {source["alias"]: source for source in manifest["sources"]}
        bodies = []
        for alias in (record["resume_alias"], record["web_scenario_alias"]):
            if alias:
                source = sources[alias]
                bodies.append((alias, source["kind"], (DATASET / source["path"]).read_text()))
    web_sources, web_evidence, documents = [], [], []
    for alias, kind, body in bodies:
        if kind == "web":
            source_id = "web-v1:" + alias
            web_sources.append(
                ResearchSourceV1(
                    source_id=source_id,
                    title=alias,
                    url="https://example.invalid/" + alias,
                    snippet=body,
                )
            )
            web_evidence.append(
                ResearchEvidenceV1(
                    evidence_id="web-evidence-v1:" + alias, source_id=source_id, text=body
                )
            )
        else:
            document_id, chunk_id = uuid4(), uuid4()
            documents.append(
                ResearchEvidenceV2(
                    source_type="workspace_document",
                    document_id=document_id,
                    chunk_id=chunk_id,
                    source_id=f"workspace-document-v1:{document_id}",
                    evidence_id=f"workspace-chunk-v1:{chunk_id}",
                    text=body,
                    ordinal=0,
                )
            )
    return EvidenceValidationNodeInputV1(
        request=ResearchRequestV1(query=query, include_application_draft=application),
        plan=ResearchPlanV1(queries=(query,)),
        sources=tuple(web_sources),
        evidence=tuple(web_evidence),
        document_evidence=tuple(documents),
        research_pass_count=1,
    )


def writer_input(node):
    return WriteReportNodeInputV1(
        request=node.request,
        evidence=node.evidence,
        document_evidence=node.document_evidence,
        evidence_sufficient=True,
    )


def writer_output(node, text):
    evidence = (*node.document_evidence, *node.evidence)[0]
    citation = ResearchCitationV1(source_id=evidence.source_id, evidence_id=evidence.evidence_id)
    claim = ResearchClaimV1(claim_id="summary", text=text, citations=(citation,))
    draft = None
    if node.request.include_application_draft:
        draft = ApplicationDraftV1(paragraphs=(claim.model_copy(update={"claim_id": "draft"}),))
    return WriteReportNodeOutputV1(summary=(claim,), application_draft=draft)


@pytest.fixture
def raw():
    return json.loads(ARTIFACT.read_bytes())


def test_reviewed_semantics_bind_immutable_sources_without_new_comparison_slots(raw):
    artifact = checked(raw)
    assert quality_identity_digest(raw) == SEMANTICS_DIGEST
    assert len({case.category for case in artifact.cases}) == 6
    plan = json.loads((ROOT / "evals/experiments/e63-evidence-sufficiency-v1.json").read_bytes())
    assert plan["samples"]["planned_slots"] == 144
    assert not set(SYNTHETIC) & set(plan["samples"]["selected_case_ids"])


@pytest.mark.parametrize("case_id", tuple(EXPECTED))
async def test_legacy_presence_characterization_is_not_candidate_semantic_accuracy(case_id, raw):
    node = runtime_input(case_id)
    old = await DeterministicEvidenceValidationNode()(node)
    registered = next(case for case in checked(raw).cases if case.case_id == case_id)
    assert old.evidence_sufficient is registered.old_evidence_sufficient is True
    # The old node never assessed relevance, completeness, conflict or claim strength.
    assert not hasattr(old, "outcome")


async def test_legacy_source_metadata_without_delivered_evidence_is_insufficient():
    node = runtime_input("office_conflict").model_copy(update={"evidence": ()})
    result = await DeterministicEvidenceValidationNode()(node)
    assert result.evidence_sufficient is False


@pytest.mark.parametrize("case_id", ["pager_shadow", "sharding_gap", "e7a_resume_only"])
def test_grounded_resume_draft_is_eligible_without_invented_job_fit(case_id):
    node = writer_input(runtime_input(case_id))
    # Verbatim supplied resume content preserves explicit responsibility/skill limits.
    output = writer_output(node, node.document_evidence[0].text)
    _validate_writer_grounding(output, node)
    assert output.application_draft is not None
    if case_id != "sharding_gap":
        assert node.evidence == ()


@pytest.mark.parametrize(
    ("case_id", "unsupported"),
    [
        ("pager_shadow", "我独立完成了生产告警排障与恢复。"),
        ("sharding_gap", "我已经负责分库分表路由维护。"),
        ("e7a_unrelated_only", "我具备数据库生产运维经验。"),
    ],
)
def test_legacy_valid_citation_does_not_check_claim_entailment(case_id, unsupported):
    node = writer_input(runtime_input(case_id))
    _validate_writer_grounding(writer_output(node, unsupported), node)
    # Acceptance here documents the old limitation; these are forbidden candidate claims.


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("missing_evidence", "citation_evidence_unknown"),
        ("wrong_source", "citation_source_mismatch"),
    ],
)
def test_legacy_citation_identity_still_rejects_unknown_or_misattributed_source(mutation, code):
    node = writer_input(runtime_input("sharding_gap"))
    output = writer_output(node, node.document_evidence[0].text)
    claim = output.summary[0]
    citation = claim.citations[0]
    change = (
        {"evidence_id": "unknown"}
        if mutation == "missing_evidence"
        else {"source_id": node.evidence[0].source_id}
    )
    bad = claim.model_copy(update={"citations": (citation.model_copy(update=change),)})
    with pytest.raises(ResearchWriterNodeError) as error:
        _validate_writer_grounding(output.model_copy(update={"summary": (bad,)}), node)
    assert error.value.schema_error_type == code


def test_legacy_nonempty_partial_report_cannot_declare_insufficiency():
    node = writer_input(runtime_input("start_unknown"))
    output = writer_output(node, node.document_evidence[0].text).model_copy(
        update={
            "limitations": (
                ResearchLimitationV1(code="insufficient_evidence", detail="Unknown date."),
            )
        }
    )
    with pytest.raises(ResearchWriterNodeError) as error:
        _validate_writer_grounding(output, node)
    assert error.value.schema_error_type == "insufficient_limitation_conflict"


@pytest.mark.parametrize("case_id", ["pager_shadow", "sharding_gap", "e7a_resume_only"])
def test_legacy_application_requires_draft_when_presence_true(case_id):
    node = writer_input(runtime_input(case_id))
    output = writer_output(node, node.document_evidence[0].text)
    with pytest.raises(ResearchWriterNodeError) as error:
        _validate_writer_grounding(output.model_copy(update={"application_draft": None}), node)
    assert error.value.schema_error_type == "application_draft_missing"


def test_gold_and_review_fields_are_not_projected_to_runtime_or_writer(monkeypatch):
    original = Path.read_bytes

    def guarded_read(path):
        assert path not in {ARTIFACT, BASELINE}
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    records = {
        row["case_id"]: row
        for row in map(json.loads, (DATASET / "cases.jsonl").read_text().splitlines())
    }
    records = copy.deepcopy(records)
    for record in records.values():
        for key in (
            "required_unit_ids",
            "expected_behavior",
            "expected_outcome",
            "tags",
            "forbidden_inferences",
            "historical_score",
            "gold",
        ):
            record[key] = "e7a-gold-review-canary"
    for case_id in EXPECTED:
        node = runtime_input(case_id, records)
        serialized = node.model_dump_json() + _writer_user_message(writer_input(node))
        forbidden = ("required_unit_ids", "expected_outcome", "forbidden_inferences", "gold")
        if "e7a-gold-review-canary" in serialized or any(
            f'"{key}"' in serialized for key in forbidden
        ):
            pytest.fail("runtime_label_projection")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_case",
        "duplicate_case",
        "missing_source",
        "foreign_source",
        "source_digest",
        "wrong_source_kind",
        "validation_case",
        "output_digest",
        "review_digest",
        "synthetic_as_observed",
        "conflict_as_failure",
        "partial_draft",
        "conflict_followup",
        "resume_blanket_refusal",
        "gap_blanket_refusal",
        "unknown_outcome",
        "body_field",
        "runtime_authorization",
        "baseline_digest",
        "package_digest",
        "candidate_pass",
    ],
)
def test_semantics_reject_missing_sources_false_provenance_or_policy_contradictions(raw, mutation):
    cases = raw["cases"]
    if mutation == "missing_case":
        cases.pop()
    elif mutation == "duplicate_case":
        cases[-1] = cases[0]
    elif mutation == "missing_source":
        cases[2]["sources"] = []
    elif mutation == "foreign_source":
        cases[2]["sources"] = cases[1]["sources"]
    elif mutation == "source_digest":
        cases[2]["sources"][0]["digest"] = "sha256:" + "0" * 64
    elif mutation == "wrong_source_kind":
        cases[2]["sources"][0]["kind"] = "web"
    elif mutation == "validation_case":
        cases[2]["source_case_id"] = "airgap_gap"
    elif mutation in {"output_digest", "review_digest"}:
        cases[2]["observed_" + mutation] = "sha256:" + "0" * 64
    elif mutation == "synthetic_as_observed":
        cases[0]["origin"] = "e4_observed"
    elif mutation == "conflict_as_failure":
        cases[3]["old_behavior"] = ["observed_overclaim"]
    elif mutation == "partial_draft":
        cases[1]["draft_eligible"] = True
    elif mutation == "conflict_followup":
        cases[3]["followup_policy"] = "retrievable_gap_once_within_shared_budget"
    elif mutation in {"resume_blanket_refusal", "gap_blanket_refusal"}:
        case = cases[4 if mutation == "resume_blanket_refusal" else 5]
        case["expected_outcome"], case["draft_eligible"] = "partial", False
    elif mutation == "unknown_outcome":
        cases[0]["expected_outcome"] = "approved"
    elif mutation == "body_field":
        cases[0]["output_text"] = "private-body-canary"
    elif mutation == "runtime_authorization":
        cases[4]["approval_granted"] = True
    elif mutation in {"baseline_digest", "package_digest"}:
        key = (
            "accepted_baseline_digest"
            if mutation == "baseline_digest"
            else "implementation_package_digest"
        )
        raw[key] = "sha256:" + "0" * 64
    else:
        raw["candidate_execution"] = "PASS"
    with pytest.raises((ValidationError, AssertionError)):
        checked(raw)
