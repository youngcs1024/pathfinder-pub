"""Gate 11.4 accepted live evidence lifecycle; eval-only, never production state."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import StringConstraints, ValidationError, model_validator

from tests.evals.contracts import EvalContractModel, EvalIdentifier
from tests.evals.live_contracts import AcceptedLiveEvalBaselineV5, LiveEvalReportV5
from tests.evals.live_retrieval_contracts import (
    AcceptedLiveRetrievalBaselineV2,
    LiveRetrievalReportV2,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIVE_BASELINE_DIRECTORY = PROJECT_ROOT / "evals" / "baselines" / "live"
GitCommitSha = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$"),
]
AcceptanceReason = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=500, pattern=r"^[^\r\n]+$"),
]


class LiveBaselineError(Exception):
    def __init__(self, category: str, *, exit_code: int = 2) -> None:
        self.category = category
        self.exit_code = exit_code
        super().__init__(category)


class LiveChatCandidateV1(EvalContractModel):
    candidate_schema_version: Literal[1] = 1
    source_commit_sha: GitCommitSha
    report: LiveEvalReportV5

    @model_validator(mode="after")
    def report_is_accepted_mode_measurement(self) -> LiveChatCandidateV1:
        if self.report.exploratory:
            raise ValueError("accepted chat candidate cannot contain exploratory evidence")
        return self


class LiveRetrievalCandidateV1(EvalContractModel):
    candidate_schema_version: Literal[1] = 1
    source_commit_sha: GitCommitSha
    report: LiveRetrievalReportV2

    @model_validator(mode="after")
    def report_is_accepted_mode_measurement(self) -> LiveRetrievalCandidateV1:
        if self.report.exploratory:
            raise ValueError("accepted retrieval candidate cannot contain exploratory evidence")
        return self


def verified_live_chat_candidate(
    *, report: LiveEvalReportV5, start_commit_sha: str, end_commit_sha: str
) -> LiveChatCandidateV1:
    if start_commit_sha != end_commit_sha:
        raise LiveBaselineError("SOURCE_COMMIT_MISMATCH")
    try:
        return LiveChatCandidateV1(source_commit_sha=start_commit_sha, report=report)
    except ValidationError:
        raise LiveBaselineError("INVALID_ACCEPTED_EVIDENCE") from None


def verified_live_retrieval_candidate(
    *, report: LiveRetrievalReportV2, start_commit_sha: str, end_commit_sha: str
) -> LiveRetrievalCandidateV1:
    if start_commit_sha != end_commit_sha:
        raise LiveBaselineError("SOURCE_COMMIT_MISMATCH")
    try:
        return LiveRetrievalCandidateV1(source_commit_sha=start_commit_sha, report=report)
    except ValidationError:
        raise LiveBaselineError("INVALID_ACCEPTED_EVIDENCE") from None


class Gate11AcceptedLiveBaselineV1(EvalContractModel):
    baseline_schema_version: Literal[1] = 1
    source_commit_sha: GitCommitSha
    acceptance_reason: AcceptanceReason
    chat: AcceptedLiveEvalBaselineV5
    retrieval: AcceptedLiveRetrievalBaselineV2

    @model_validator(mode="after")
    def inner_measurements_are_strict_accepted_evidence(
        self,
    ) -> Gate11AcceptedLiveBaselineV1:
        AcceptedLiveEvalBaselineV5.model_validate_json(self.chat.model_dump_json(), strict=True)
        AcceptedLiveRetrievalBaselineV2.model_validate_json(
            self.retrieval.model_dump_json(), strict=True
        )
        if self.acceptance_reason != self.acceptance_reason.strip():
            raise ValueError("acceptance reason must be normalized and nonblank")
        chat = self.chat.report
        retrieval = self.retrieval.report
        if (
            chat.artifact_identity.embedding_profile != retrieval.embedding_profile
            or chat.version_metadata.embedding_model != "text-embedding-v4"
            or chat.version_metadata.embedding_dimension != retrieval.embedding_dimension
        ):
            raise ValueError("chat and retrieval evidence profiles must be consistent")
        return self


class LiveMetricDeltaV1(EvalContractModel):
    metric: EvalIdentifier
    baseline: Decimal | None
    current: Decimal | None
    delta: Decimal | None
    availability: Literal["available", "unknown_cost", "not_recorded"] = "available"


class Gate11LiveComparisonV1(EvalContractModel):
    schema_version: Literal[1] = 1
    status: Literal["COMPARABLE", "NEW_BASELINE_REQUIRED"]
    identity_changes: tuple[str, ...] = ()
    metrics: tuple[LiveMetricDeltaV1, ...] = ()
    cost_comparison_available: bool
    interpretation: Literal["human_review_diagnostic_no_automatic_percentage_gate"] = (
        "human_review_diagnostic_no_automatic_percentage_gate"
    )


def probe_clean_git_head(
    *,
    project_root: Path = PROJECT_ROOT,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    try:
        head = runner(
            ("git", "rev-parse", "HEAD"),
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        status = runner(
            ("git", "status", "--porcelain"),
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        raise LiveBaselineError("GIT_PROBE_FAILED") from None
    commit_sha = head.stdout.strip()
    if head.returncode or not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise LiveBaselineError("INVALID_SOURCE_COMMIT")
    if status.returncode:
        raise LiveBaselineError("GIT_PROBE_FAILED")
    if status.stdout:
        raise LiveBaselineError("DIRTY_WORKTREE")
    return commit_sha


def prepare_accepted_live_baseline(
    *,
    chat_candidate: LiveChatCandidateV1,
    retrieval_candidate: LiveRetrievalCandidateV1,
    acceptance_reason: str,
    git_probe: Callable[[], str] = probe_clean_git_head,
) -> Gate11AcceptedLiveBaselineV1:
    try:
        commit_sha = git_probe()
        if not (
            chat_candidate.source_commit_sha == retrieval_candidate.source_commit_sha == commit_sha
        ):
            raise LiveBaselineError("SOURCE_COMMIT_MISMATCH")
        return Gate11AcceptedLiveBaselineV1(
            source_commit_sha=commit_sha,
            acceptance_reason=acceptance_reason,
            chat=AcceptedLiveEvalBaselineV5(report=chat_candidate.report),
            retrieval=AcceptedLiveRetrievalBaselineV2(report=retrieval_candidate.report),
        )
    except LiveBaselineError:
        raise
    except (ValidationError, ValueError):
        raise LiveBaselineError("INVALID_ACCEPTED_EVIDENCE") from None


def accepted_live_baseline_path(
    baseline: Gate11AcceptedLiveBaselineV1,
    directory: Path = DEFAULT_LIVE_BASELINE_DIRECTORY,
) -> Path:
    return directory / f"gate11-live-v1-{baseline.source_commit_sha}.json"


def persist_accepted_live_baseline(
    baseline: Gate11AcceptedLiveBaselineV1,
    directory: Path = DEFAULT_LIVE_BASELINE_DIRECTORY,
    *,
    before_publish: Callable[[Path, Path], None] | None = None,
    link_operation: Callable[[str, str], None] = os.link,
    cleanup_operation: Callable[[str], None] = os.unlink,
) -> Path:
    try:
        validated = Gate11AcceptedLiveBaselineV1.model_validate_json(
            baseline.model_dump_json(), strict=True
        )
    except ValidationError:
        raise LiveBaselineError("INVALID_ACCEPTED_EVIDENCE") from None
    final_path = accepted_live_baseline_path(validated, directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise LiveBaselineError("BASELINE_WRITE_FAILED") from None
    if final_path.exists():
        raise LiveBaselineError("BASELINE_EXISTS")

    temporary_name: str | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="strict",
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(validated.model_dump_json(indent=2) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        if before_publish is not None:
            before_publish(Path(temporary_name), final_path)
        try:
            link_operation(temporary_name, str(final_path))
        except FileExistsError:
            raise LiveBaselineError("BASELINE_EXISTS") from None
        published = True
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except LiveBaselineError:
        raise
    except OSError:
        raise LiveBaselineError("BASELINE_WRITE_FAILED") from None
    finally:
        if temporary_name is not None:
            try:
                cleanup_operation(temporary_name)
            except OSError:
                # Once published, the final hard link is complete and authoritative.
                # Before publication, a non-authoritative dot-temp may remain.
                pass
    if not published:
        raise LiveBaselineError("BASELINE_WRITE_FAILED")
    return final_path


def load_accepted_live_baseline(path: Path) -> Gate11AcceptedLiveBaselineV1:
    try:
        serialized = path.read_text(encoding="utf-8", errors="strict")
    except FileNotFoundError:
        raise LiveBaselineError("BASELINE_MISSING") from None
    except (OSError, UnicodeError):
        raise LiveBaselineError("BASELINE_UNREADABLE") from None
    try:
        payload = json.loads(serialized)
    except (json.JSONDecodeError, UnicodeError):
        raise LiveBaselineError("BASELINE_MALFORMED_JSON") from None
    if not isinstance(payload, dict) or payload.get("baseline_schema_version") != 1:
        raise LiveBaselineError("BASELINE_SCHEMA_UNSUPPORTED")
    try:
        return Gate11AcceptedLiveBaselineV1.model_validate_json(serialized, strict=True)
    except ValidationError:
        raise LiveBaselineError("BASELINE_SCHEMA_INCOMPATIBLE") from None


def _decimal(value: int | float | Decimal | None) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def _delta(
    metric: str,
    baseline: int | float | Decimal | None,
    current: int | float | Decimal | None,
    *,
    availability: Literal["available", "unknown_cost", "not_recorded"] = "available",
) -> LiveMetricDeltaV1:
    old, new = _decimal(baseline), _decimal(current)
    return LiveMetricDeltaV1(
        metric=metric,
        baseline=old,
        current=new,
        delta=new - old
        if old is not None and new is not None and availability == "available"
        else None,
        availability=availability,
    )


def _identity_changes(
    baseline: Gate11AcceptedLiveBaselineV1,
    current: Gate11AcceptedLiveBaselineV1,
) -> tuple[str, ...]:
    old_chat, new_chat = baseline.chat.report, current.chat.report
    old_retrieval, new_retrieval = baseline.retrieval.report, current.retrieval.report
    fields = (
        ("chat_manifest", old_chat.manifest_digest, new_chat.manifest_digest),
        ("chat_artifact", old_chat.artifact_identity, new_chat.artifact_identity),
        ("chat_versions", old_chat.version_metadata, new_chat.version_metadata),
        ("retrieval_dataset", old_retrieval.dataset_digest, new_retrieval.dataset_digest),
        ("retrieval_case_set", old_retrieval.case_set_digest, new_retrieval.case_set_digest),
        (
            "retrieval_normalization",
            old_retrieval.normalization_version,
            new_retrieval.normalization_version,
        ),
        ("retrieval_chunking", old_retrieval.chunking_version, new_retrieval.chunking_version),
        ("retrieval_embedding", old_retrieval.embedding_profile, new_retrieval.embedding_profile),
        ("retrieval_top_k", old_retrieval.retrieval_top_k, new_retrieval.retrieval_top_k),
    )
    return tuple(name for name, old, new in fields if old != new)


def compare_accepted_live_baselines(
    baseline: Gate11AcceptedLiveBaselineV1,
    current: Gate11AcceptedLiveBaselineV1,
) -> Gate11LiveComparisonV1:
    changes = _identity_changes(baseline, current)
    old_chat, new_chat = baseline.chat.report.aggregate, current.chat.report.aggregate
    old_ret, new_ret = baseline.retrieval.report, current.retrieval.report
    cost_available = not (
        old_chat.unknown_cost_attempt_count
        or new_chat.unknown_cost_attempt_count
        or old_ret.unknown_cost_attempt_count
        or new_ret.unknown_cost_attempt_count
    )
    if changes:
        return Gate11LiveComparisonV1(
            status="NEW_BASELINE_REQUIRED",
            identity_changes=changes,
            cost_comparison_available=cost_available,
        )
    cost_availability = "available" if cost_available else "unknown_cost"
    metrics = (
        _delta("chat_case_pass_rate", old_chat.case_pass_rate, new_chat.case_pass_rate),
        _delta(
            "chat_structured_valid_rate",
            old_chat.structured_output_valid_rate,
            new_chat.structured_output_valid_rate,
        ),
        _delta(
            "chat_tool_schema_valid_rate",
            old_chat.tool_argument_schema_valid_rate,
            new_chat.tool_argument_schema_valid_rate,
        ),
        _delta(
            "chat_logical_model_calls", old_chat.logical_model_calls, new_chat.logical_model_calls
        ),
        _delta("chat_provider_attempts", old_chat.provider_attempts, new_chat.provider_attempts),
        _delta("chat_input_tokens", old_chat.input_tokens, new_chat.input_tokens),
        _delta("chat_output_tokens", old_chat.output_tokens, new_chat.output_tokens),
        _delta(
            "chat_reasoning_tokens",
            old_chat.reasoning_tokens,
            new_chat.reasoning_tokens,
            availability="available"
            if old_chat.reasoning_tokens is not None and new_chat.reasoning_tokens is not None
            else "not_recorded",
        ),
        _delta(
            "chat_known_cost_cny",
            old_chat.known_cost_cny,
            new_chat.known_cost_cny,
            availability=cost_availability,
        ),
        _delta(
            "chat_unknown_cost_attempts",
            old_chat.unknown_cost_attempt_count,
            new_chat.unknown_cost_attempt_count,
        ),
        _delta(
            "chat_case_latency_p50",
            old_chat.case_latency.observed_p50_ms,
            new_chat.case_latency.observed_p50_ms,
        ),
        _delta(
            "chat_case_latency_p95",
            old_chat.case_latency.observed_p95_ms,
            new_chat.case_latency.observed_p95_ms,
        ),
        _delta(
            "chat_provider_latency_p50",
            old_chat.provider_latency.observed_p50_ms,
            new_chat.provider_latency.observed_p50_ms,
        ),
        _delta(
            "chat_provider_latency_p95",
            old_chat.provider_latency.observed_p95_ms,
            new_chat.provider_latency.observed_p95_ms,
        ),
        _delta(
            "retrieval_recall_at_1",
            old_ret.aggregate.macro_recall_at_1,
            new_ret.aggregate.macro_recall_at_1,
        ),
        _delta(
            "retrieval_recall_at_3",
            old_ret.aggregate.macro_recall_at_3,
            new_ret.aggregate.macro_recall_at_3,
        ),
        _delta(
            "retrieval_recall_at_5",
            old_ret.aggregate.macro_recall_at_5,
            new_ret.aggregate.macro_recall_at_5,
        ),
        _delta(
            "retrieval_mrr",
            old_ret.aggregate.mean_reciprocal_rank,
            new_ret.aggregate.mean_reciprocal_rank,
        ),
        _delta(
            "retrieval_irrelevant_context",
            old_ret.aggregate.total_irrelevant_context_count,
            new_ret.aggregate.total_irrelevant_context_count,
        ),
        _delta(
            "retrieval_provider_attempts",
            old_ret.provider_attempt_count,
            new_ret.provider_attempt_count,
        ),
        _delta("retrieval_input_tokens", old_ret.input_tokens, new_ret.input_tokens),
        _delta(
            "retrieval_known_cost_cny",
            old_ret.known_cost_cny,
            new_ret.known_cost_cny,
            availability=cost_availability,
        ),
        _delta(
            "retrieval_unknown_cost_attempts",
            old_ret.unknown_cost_attempt_count,
            new_ret.unknown_cost_attempt_count,
        ),
        _delta(
            "retrieval_provider_latency_p50",
            old_ret.embedding_provider_latency.observed_p50_ms,
            new_ret.embedding_provider_latency.observed_p50_ms,
        ),
        _delta(
            "retrieval_provider_latency_p95",
            old_ret.embedding_provider_latency.observed_p95_ms,
            new_ret.embedding_provider_latency.observed_p95_ms,
        ),
    )
    return Gate11LiveComparisonV1(
        status="COMPARABLE",
        metrics=metrics,
        cost_comparison_available=cost_available,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Accept immutable Gate 11 live evidence.")
    parser.add_argument("--chat-report", type=Path, required=True)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_LIVE_BASELINE_DIRECTORY)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        chat = LiveChatCandidateV1.model_validate_json(parsed.chat_report.read_bytes(), strict=True)
        retrieval = LiveRetrievalCandidateV1.model_validate_json(
            parsed.retrieval_report.read_bytes(), strict=True
        )
        accepted = prepare_accepted_live_baseline(
            chat_candidate=chat,
            retrieval_candidate=retrieval,
            acceptance_reason=parsed.reason,
        )
        path = persist_accepted_live_baseline(accepted, parsed.output_directory)
    except (OSError, ValidationError):
        error = LiveBaselineError("INVALID_ACCEPTANCE_INPUT")
    except LiveBaselineError as caught:
        error = caught
    else:
        sys.stdout.write(accepted.model_dump_json(indent=2) + "\n")
        sys.stderr.write(f"LIVE BASELINE ACCEPTED {path.name}\n")
        return 0
    sys.stderr.write(f"LIVE BASELINE ERROR {error.category}\n")
    return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
