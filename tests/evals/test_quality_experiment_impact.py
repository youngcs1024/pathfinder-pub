"""Impact integrity and honest dev-only boundaries, not candidate behavior evidence."""

import io
import json
import socket
import subprocess
from pathlib import Path

import pytest

from tests.evals import quality_experiment_impact as impact

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "evals/experiments/e64-evidence-sufficiency-impact-v1.json"
CANARY = "private-impact-error-canary"


@pytest.fixture
def raw():
    return json.loads(REVIEW.read_bytes())


def write_review(tmp_path, raw):
    target = tmp_path / "review.json"
    target.write_text(json.dumps(raw))
    return target


def test_registration_preserves_baseline_and_distinguishes_candidate_from_production():
    review = impact.load_impact_review(REVIEW)
    assert review.reviewed_source_sha == "cbe8faf383e5a9fe72730d33128cc2c10c733b68"
    assert review.baseline_source_sha == "6ee09703fd19dd3f68813bb40f2b314fba512995"
    assert review.candidate_graph_version != review.reviewed_production_graph_version
    assert review.reviewed_output_versions == (1, 2)
    assert review.output_policy.sufficient == "grounded_valid_draft_eligible_not_authorized"
    assert review.database.result_authority == "test_postgresql"
    assert review.database.both_arms == "same_fixture_schema"
    assert review.database.production_migration == "none"
    assert (review.budget.model_calls_per_run, review.budget.tool_calls_per_run) == (12, 8)
    assert review.budget.research_passes_per_run == 2
    assert len(review.impacts) == 10
    assert review.execution_readiness == "NOT_VERIFIED"
    assert review.candidate_execution == review.adoption == review.deployment == "NOT_RUN"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("reviewed_source_sha", "a" * 40),
        ("baseline_source_sha", "b" * 40),
        ("baseline_src_tree", "c" * 40),
        ("plan_digest", "sha256:" + "d" * 64),
        ("plan_path", "evals/experiments/another.json"),
        ("selected_branch", "B"),
        ("experiment_only", False),
        ("experiment_only", 1),
        ("candidate_location", "src/app/agents"),
        ("candidate_graph_version", "pathfinder-research-v6"),
        ("candidate_output_contract", "ResearchOutputV2"),
        ("reviewed_database_revision", "new_revision"),
        ("execution_readiness", "PASS"),
        ("candidate_execution", "PASS"),
        ("adoption", "PASS"),
        ("deployment", "PASS"),
    ],
)
def test_identity_scope_and_unexecuted_claims_cannot_be_changed(raw, tmp_path, field, replacement):
    raw[field] = replacement
    with pytest.raises(impact.ImpactReviewError):
        impact.load_impact_review(write_review(tmp_path, raw))


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("database", "admission", "any_loopback_dsn"),
        ("database", "schema_change", "disable_all_constraints"),
        ("database", "both_arms", "candidate_only"),
        ("database", "business_enums", "add_experiment_status"),
        ("database", "result_authority", "private_json_only"),
        ("database", "result_reader", "production_api"),
        ("database", "production_migration", "required_now"),
        ("database", "new_tables", True),
        ("output_policy", "sufficient", "automatically_authorized"),
        ("output_policy", "partial", "allow_action"),
        ("output_policy", "insufficient", "allow_action"),
        ("output_policy", "conflicting", "choose_source_and_submit"),
        ("output_policy", "resume_only", "always_require_web"),
        ("output_policy", "technical_failure", "return_insufficient"),
        ("budget", "model_calls_per_run", 13),
        ("budget", "tool_calls_per_run", 9),
        ("budget", "research_passes_per_run", 3),
        ("budget", "assessor_calls_per_pass", 2),
        ("budget", "allocation", "extra_free_calls"),
        ("budget", "model_calls_per_run", 12.0),
    ],
)
def test_database_action_and_shared_budget_boundaries_cannot_be_relaxed(
    raw, tmp_path, section, field, replacement
):
    raw[section][field] = replacement
    with pytest.raises(impact.ImpactReviewError):
        impact.load_impact_review(write_review(tmp_path, raw))


@pytest.mark.parametrize(
    "mutation",
    ["missing_area", "duplicate_area", "unknown_area", "disposition", "requirement", "source"],
)
def test_impact_coverage_and_source_mapping_are_mandatory(raw, tmp_path, mutation):
    row = raw["impacts"][0]
    if mutation == "missing_area":
        raw["impacts"].pop()
    elif mutation == "duplicate_area":
        raw["impacts"][-1] = row
    elif mutation == "unknown_area":
        row["area"] = "new_algorithm"
    elif mutation == "disposition":
        row["disposition"] = "dev_only"
    elif mutation == "requirement":
        row["requirements"].pop()
    else:
        row["reviewed_files"][0] = "src/app/main.py"
    with pytest.raises(impact.ImpactReviewError):
        impact.load_impact_review(write_review(tmp_path, raw))


@pytest.mark.parametrize("section", ["adoption_requirements", "e65_handoff"])
def test_required_followup_cannot_be_silently_omitted(raw, tmp_path, section):
    raw[section].pop()
    with pytest.raises(impact.ImpactReviewError):
        impact.load_impact_review(write_review(tmp_path, raw))


@pytest.mark.parametrize(
    "name", ["../outside", "/tmp/outside", "src//app/main.py", "src/./app/main.py", "C:\\x", ""]
)
def test_untrusted_reference_paths_are_rejected_before_inspection(raw, tmp_path, monkeypatch, name):
    raw["impacts"][0]["reviewed_files"][0] = name
    target = write_review(tmp_path, raw)
    original = Path.is_file

    def checked(path):
        assert path.is_relative_to(ROOT), "reference escaped the repository"
        return original(path)

    monkeypatch.setattr(Path, "is_file", checked)
    with pytest.raises(impact.ImpactReviewError, match=r"^invalid_impact_path$"):
        impact.load_impact_review(target)


@pytest.mark.parametrize("directory_link", [False, True])
def test_symlinked_reference_cannot_escape_root(tmp_path, directory_link):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.py").write_text(CANARY)
    root = tmp_path / "root"
    root.mkdir()
    if directory_link:
        (root / "link").symlink_to(outside, target_is_directory=True)
        name = "link/source.py"
    else:
        (root / "link.py").symlink_to(outside / "source.py")
        name = "link.py"
    with pytest.raises(impact.ImpactReviewError, match=r"^invalid_impact_path$"):
        impact._public_file(root, name)


@pytest.mark.parametrize("resource", ["plan", "accepted", "lock"])
def test_actual_bound_source_tampering_is_rejected(monkeypatch, resource):
    target = (
        ROOT
        / {
            "plan": impact.PLAN_PATH,
            "accepted": "evals/baselines/quality/e410-agent-v1.json",
            "lock": "uv.lock",
        }[resource]
    )
    original = Path.open
    injected = []

    def changed(path, *args, **kwargs):
        if path == target:
            injected.append(path)
            return io.BytesIO(b"{}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changed)
    with pytest.raises(impact.ImpactReviewError):
        impact.load_impact_review(REVIEW)
    assert injected == [target]


def test_missing_reviewed_source_fails_closed(monkeypatch):
    target = ROOT / "src/app/retrieval/chunking.py"
    original = Path.is_file
    inspected = []

    def missing(path):
        if path == target:
            inspected.append(path)
            return False
        return original(path)

    monkeypatch.setattr(Path, "is_file", missing)
    with pytest.raises(impact.ImpactReviewError, match=r"^invalid_impact_path$"):
        impact.load_impact_review(REVIEW)
    assert inspected == [target]


def test_model_copy_does_not_bypass_validation():
    review = impact.load_impact_review(REVIEW)
    forged = review.model_copy(update={"experiment_only": False})
    with pytest.raises(impact.ImpactReviewError):
        impact.validate_impact_review(forged)


@pytest.mark.parametrize("mutation", ["root", "nested", "duplicate", "oversized"])
def test_unknown_fields_duplicate_keys_and_oversized_json_are_rejected(raw, tmp_path, mutation):
    if mutation == "root":
        raw[CANARY] = CANARY
    elif mutation == "nested":
        raw["database"][CANARY] = CANARY
    target = write_review(tmp_path, raw)
    if mutation == "duplicate":
        target.write_text('{"review_id":"first","review_id":"second"}')
    elif mutation == "oversized":
        target.write_bytes(b" " * (impact.MAX_REVIEW_BYTES + 1))
    with pytest.raises(impact.ImpactReviewError, match=r"^invalid_impact_review$"):
        impact.load_impact_review(target)


def test_cli_reads_only_public_files_without_network_processes_or_writes(monkeypatch, capsys):
    original = Path.open
    reads = []

    def public_read(path, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+"), "validation attempted a write"
        assert path.is_relative_to(ROOT)
        relative = path.relative_to(ROOT)
        assert relative.parts[0] not in {"docs", ".git"}
        reads.append(relative.as_posix())
        return original(path, mode, *args, **kwargs)

    def prohibited(*args, **kwargs):
        pytest.fail("validation attempted an external action")

    monkeypatch.setattr(Path, "open", public_read)
    monkeypatch.setattr(socket, "create_connection", prohibited)
    monkeypatch.setattr(socket.socket, "connect", prohibited)
    monkeypatch.setattr(subprocess, "Popen", prohibited)
    assert impact.main(["validate", "--review", str(REVIEW)]) == 0
    result = capsys.readouterr()
    output = json.loads(result.out)
    assert output["category"] == "impact_review_valid"
    assert output["verification_limit"] == "impact_registration_only"
    assert output["execution_readiness"] == "NOT_VERIFIED"
    assert output["candidate_execution"] == output["adoption"] == output["deployment"] == "NOT_RUN"
    assert impact.PLAN_PATH in reads and "uv.lock" in reads
    assert not result.err and CANARY not in result.out


@pytest.mark.parametrize("kind", ["arguments", "path", "json", "schema", "source", "exception"])
def test_cli_never_echoes_private_content(raw, tmp_path, monkeypatch, capsys, kind):
    target = tmp_path / CANARY
    args = ["validate", "--review", str(target)]
    if kind == "arguments":
        args = ["execute", CANARY]
    elif kind == "json":
        target.write_text(CANARY)
    elif kind == "schema":
        raw["database"]["admission"] = CANARY
        target.write_text(json.dumps(raw))
    elif kind in {"source", "exception"}:
        args = ["validate", "--review", str(REVIEW)]

        def fail(*args, **kwargs):
            if kind == "exception":
                raise impact.ImpactReviewError(CANARY)
            raise OSError(CANARY)

        monkeypatch.setattr(impact, "load_impact_review", fail)
    assert impact.main(args) == 1
    result = capsys.readouterr()
    assert set(json.loads(result.out)) == {"category"}
    assert CANARY not in result.out + result.err
    assert not result.err
