from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.evals.live_baseline import (
    Gate11AcceptedLiveBaselineV1,
    LiveBaselineError,
    LiveChatCandidateV1,
    LiveRetrievalCandidateV1,
    compare_accepted_live_baselines,
    load_accepted_live_baseline,
    persist_accepted_live_baseline,
    prepare_accepted_live_baseline,
    probe_clean_git_head,
    verified_live_chat_candidate,
    verified_live_retrieval_candidate,
)
from tests.evals.live_retrieval_contracts import LiveRetrievalReportV2
from tests.evals.test_live_retrieval_contract import _report_payload
from tests.evals.test_live_suite_contract import report as accepted_chat_report


def _retrieval_report() -> LiveRetrievalReportV2:
    payload = _report_payload()
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
    return LiveRetrievalReportV2.model_validate_json(json.dumps(payload), strict=True)


def _baseline(commit: str = "a" * 40) -> Gate11AcceptedLiveBaselineV1:
    return prepare_accepted_live_baseline(
        chat_candidate=LiveChatCandidateV1(source_commit_sha=commit, report=accepted_chat_report()),
        retrieval_candidate=LiveRetrievalCandidateV1(
            source_commit_sha=commit, report=_retrieval_report()
        ),
        acceptance_reason="operator reviewed fixed live evidence",
        git_probe=lambda: commit,
    )


def _git_result(returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(("git",), returncode, stdout=stdout, stderr="")


def test_git_probe_accepts_clean_head_and_rejects_dirty_or_malformed():
    clean_results = iter((_git_result(0, "a" * 40 + "\n"), _git_result(0, "")))
    assert probe_clean_git_head(runner=lambda *args, **kwargs: next(clean_results)) == "a" * 40

    dirty_results = iter((_git_result(0, "a" * 40), _git_result(0, " M file.py\n")))
    with pytest.raises(LiveBaselineError, match="DIRTY_WORKTREE"):
        probe_clean_git_head(runner=lambda *args, **kwargs: next(dirty_results))

    malformed = iter((_git_result(0, "short"), _git_result(0, "")))
    with pytest.raises(LiveBaselineError, match="INVALID_SOURCE_COMMIT"):
        probe_clean_git_head(runner=lambda *args, **kwargs: next(malformed))


def test_candidate_contracts_are_strict_non_exploratory_and_roundtrip():
    commit = "a" * 40
    chat = LiveChatCandidateV1(source_commit_sha=commit, report=accepted_chat_report())
    retrieval = LiveRetrievalCandidateV1(source_commit_sha=commit, report=_retrieval_report())

    assert LiveChatCandidateV1.model_validate_json(chat.model_dump_json(), strict=True) == chat
    assert (
        LiveRetrievalCandidateV1.model_validate_json(retrieval.model_dump_json(), strict=True)
        == retrieval
    )
    for invalid_sha in ("a" * 39, "A" * 40, 1):
        with pytest.raises(ValidationError):
            LiveChatCandidateV1(source_commit_sha=invalid_sha, report=accepted_chat_report())
    with pytest.raises(ValidationError, match="exploratory"):
        LiveChatCandidateV1(
            source_commit_sha=commit,
            report=accepted_chat_report().model_copy(update={"exploratory": True}),
        )
    with pytest.raises(ValidationError, match="exploratory"):
        LiveRetrievalCandidateV1(
            source_commit_sha=commit,
            report=_retrieval_report().model_copy(update={"exploratory": True}),
        )


@pytest.mark.parametrize(
    ("chat_commit", "retrieval_commit", "head_commit"),
    [
        ("a" * 40, "b" * 40, "b" * 40),
        ("a" * 40, "a" * 40, "b" * 40),
        ("b" * 40, "a" * 40, "b" * 40),
    ],
)
def test_final_acceptance_rejects_any_source_commit_mismatch(
    chat_commit,
    retrieval_commit,
    head_commit,  # type: ignore[no-untyped-def]
):
    with pytest.raises(LiveBaselineError, match="SOURCE_COMMIT_MISMATCH"):
        prepare_accepted_live_baseline(
            chat_candidate=LiveChatCandidateV1(
                source_commit_sha=chat_commit, report=accepted_chat_report()
            ),
            retrieval_candidate=LiveRetrievalCandidateV1(
                source_commit_sha=retrieval_commit, report=_retrieval_report()
            ),
            acceptance_reason="reviewed",
            git_probe=lambda: head_commit,
        )


def test_candidate_pre_post_commit_identity_is_fail_closed():
    commit = "a" * 40
    chat = verified_live_chat_candidate(
        report=accepted_chat_report(), start_commit_sha=commit, end_commit_sha=commit
    )
    retrieval = verified_live_retrieval_candidate(
        report=_retrieval_report(), start_commit_sha=commit, end_commit_sha=commit
    )
    assert chat.source_commit_sha == retrieval.source_commit_sha == commit

    with pytest.raises(LiveBaselineError, match="SOURCE_COMMIT_MISMATCH"):
        verified_live_chat_candidate(
            report=accepted_chat_report(),
            start_commit_sha=commit,
            end_commit_sha="b" * 40,
        )
    with pytest.raises(LiveBaselineError, match="SOURCE_COMMIT_MISMATCH"):
        verified_live_retrieval_candidate(
            report=_retrieval_report(),
            start_commit_sha=commit,
            end_commit_sha="b" * 40,
        )


def test_quality_failed_candidate_is_reviewable_but_not_acceptable():
    report = accepted_chat_report()
    failed = tuple(
        observation.model_copy(
            update={
                "grader": observation.grader.model_copy(update={"citation_grounding_proxy": False})
            }
        )
        if index < 5
        else observation
        for index, observation in enumerate(report.observations)
    )
    failed_report = accepted_chat_report(failed)
    candidate = LiveChatCandidateV1(source_commit_sha="a" * 40, report=failed_report)

    assert (
        LiveChatCandidateV1.model_validate_json(candidate.model_dump_json(), strict=True)
        == candidate
    )
    with pytest.raises(LiveBaselineError, match="INVALID_ACCEPTED_EVIDENCE"):
        prepare_accepted_live_baseline(
            chat_candidate=candidate,
            retrieval_candidate=LiveRetrievalCandidateV1(
                source_commit_sha="a" * 40, report=_retrieval_report()
            ),
            acceptance_reason="reviewed failed candidate",
            git_probe=lambda: "a" * 40,
        )


@pytest.mark.parametrize(("end_commit", "expected_exit"), [("a" * 40, 0), ("b" * 40, 2)])
def test_accepted_chat_entrypoint_binds_pre_post_commit(
    monkeypatch,
    capsys,
    end_commit,
    expected_exit,  # type: ignore[no-untyped-def]
):
    import tests.evals.__main__ as eval_main
    import tests.evals.live_baseline as live_baseline
    import tests.evals.live_chat as live_chat

    commits = iter(("a" * 40, end_commit))

    async def scripted_run_live_chat(*, accepted):  # type: ignore[no-untyped-def]
        assert accepted is True
        return SimpleNamespace(report=accepted_chat_report(), exit_code=0, stop_reason=None)

    monkeypatch.delenv("PF_EVAL_CASE", raising=False)
    monkeypatch.setattr(live_baseline, "probe_clean_git_head", lambda: next(commits))
    monkeypatch.setattr(live_chat, "run_live_chat", scripted_run_live_chat)

    assert eval_main.main(["--live-chat-accepted"]) == expected_exit
    output = capsys.readouterr()
    if expected_exit == 0:
        candidate = LiveChatCandidateV1.model_validate_json(output.out, strict=True)
        assert candidate.source_commit_sha == "a" * 40
    else:
        assert output.out == ""
        assert output.err == "LIVE_CHAT FAIL SOURCE_COMMIT_MISMATCH\n"


def test_prepare_strict_load_and_immutable_create(tmp_path):
    baseline = _baseline()
    path = persist_accepted_live_baseline(baseline, tmp_path)
    original = path.read_bytes()

    assert load_accepted_live_baseline(path) == baseline
    with pytest.raises(LiveBaselineError, match="BASELINE_EXISTS"):
        persist_accepted_live_baseline(baseline, tmp_path)
    assert path.read_bytes() == original


def test_failure_after_temp_fsync_does_not_publish(tmp_path):
    baseline = _baseline()

    def fail_before_publish(_temporary, _final):  # type: ignore[no-untyped-def]
        raise OSError("injected")

    with pytest.raises(LiveBaselineError, match="BASELINE_WRITE_FAILED"):
        persist_accepted_live_baseline(baseline, tmp_path, before_publish=fail_before_publish)
    assert not any(path.name.startswith("gate11-live-") for path in tmp_path.iterdir())


def test_cleanup_failure_preserves_loadable_final(tmp_path):
    baseline = _baseline()

    def fail_cleanup(_path: str) -> None:
        raise OSError("injected")

    path = persist_accepted_live_baseline(baseline, tmp_path, cleanup_operation=fail_cleanup)
    assert load_accepted_live_baseline(path) == baseline


def test_concurrent_create_publishes_exactly_once_without_overwrite(tmp_path):
    baseline = _baseline()

    def create():  # type: ignore[no-untyped-def]
        try:
            return persist_accepted_live_baseline(baseline, tmp_path)
        except LiveBaselineError as error:
            return error.category

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _: create(), range(2)))
    assert sum(not isinstance(outcome, str) for outcome in outcomes) == 1
    assert outcomes.count("BASELINE_EXISTS") == 1
    final_paths = tuple(tmp_path.glob("gate11-live-v1-*.json"))
    assert len(final_paths) == 1
    assert load_accepted_live_baseline(final_paths[0]) == baseline


@pytest.mark.parametrize(
    ("contents", "category"),
    [
        ("{", "BASELINE_MALFORMED_JSON"),
        ('{"baseline_schema_version":99}', "BASELINE_SCHEMA_UNSUPPORTED"),
        ('{"baseline_schema_version":1}', "BASELINE_SCHEMA_INCOMPATIBLE"),
    ],
)
def test_strict_loader_error_categories(tmp_path, contents, category):  # type: ignore[no-untyped-def]
    path = tmp_path / "gate11-live-v1-a.json"
    path.write_text(contents)
    with pytest.raises(LiveBaselineError, match=category):
        load_accepted_live_baseline(path)


def test_comparison_is_diagnostic_and_identity_drift_requires_new_baseline():
    baseline = _baseline("a" * 40)
    current = _baseline("b" * 40)
    comparison = compare_accepted_live_baselines(baseline, current)
    assert comparison.status == "COMPARABLE"
    assert comparison.identity_changes == ()
    assert {metric.metric for metric in comparison.metrics} >= {
        "chat_case_pass_rate",
        "chat_provider_latency_p95",
        "retrieval_recall_at_5",
        "retrieval_provider_latency_p95",
    }

    changed_report = current.retrieval.report.model_copy(
        update={"embedding_profile": "future-profile"}
    )
    changed = current.model_copy(
        update={"retrieval": current.retrieval.model_copy(update={"report": changed_report})}
    )
    drift = compare_accepted_live_baselines(baseline, changed)
    assert drift.status == "NEW_BASELINE_REQUIRED"
    assert drift.identity_changes == ("retrieval_embedding",)
    assert drift.metrics == ()
