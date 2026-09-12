from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence

from tests.evals.harness import run_evaluation_sync


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic Pathfinder research evals.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live-chat", action="store_true", help="Run manual Qwen fixed-evidence chat eval."
    )
    mode.add_argument(
        "--live-chat-accepted",
        action="store_true",
        help="Run accepted-capable Qwen chat eval after clean-commit verification.",
    )
    mode.add_argument("--case", dest="case_id", help="Run one versioned eval case.")
    mode.add_argument(
        "--live-smoke",
        action="store_true",
        help="Run the explicit Qwen/Tavily/Langfuse/PostgreSQL smoke check.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    environment_case = os.environ.get("PF_EVAL_CASE") or None
    if parsed.live_chat or parsed.live_chat_accepted:
        from tests.evals.live_baseline import (
            LiveBaselineError,
            probe_clean_git_head,
            verified_live_chat_candidate,
        )
        from tests.evals.live_chat import run_live_chat
        from tests.evals.live_suite import LiveSuiteConfigurationError

        if environment_case is not None:
            sys.stderr.write("LIVE_CHAT FAIL conflicting_arguments\n")
            return 2
        try:
            start_commit_sha = probe_clean_git_head() if parsed.live_chat_accepted else None
            result = asyncio.run(run_live_chat(accepted=parsed.live_chat_accepted))
            if parsed.live_chat_accepted:
                assert start_commit_sha is not None
                candidate = verified_live_chat_candidate(
                    report=result.report,
                    start_commit_sha=start_commit_sha,
                    end_commit_sha=probe_clean_git_head(),
                )
                output = candidate.model_dump_json(indent=2)
            else:
                output = result.report.model_dump_json(indent=2)
        except LiveSuiteConfigurationError:
            sys.stderr.write("LIVE_CHAT FAIL invalid_configuration\n")
            return 2
        except LiveBaselineError as error:
            sys.stderr.write(f"LIVE_CHAT FAIL {error.category}\n")
            return error.exit_code
        sys.stdout.write(output + "\n")
        status = "PASS" if result.exit_code == 0 else "FAIL"
        evidence_kind = "accepted_candidate" if parsed.live_chat_accepted else "exploratory"
        sys.stderr.write(f"LIVE_CHAT {status} {result.stop_reason or evidence_kind + '_quality'}\n")
        return result.exit_code
    if parsed.live_smoke:
        from tests.evals.live_contracts import LiveSmokeReportV1
        from tests.evals.live_smoke import run_live_smoke_sync

        if environment_case is not None:
            report = LiveSmokeReportV1(
                passed=False,
                error_category="conflicting_arguments",
            )
            exit_code = 2
            missing_variables: tuple[str, ...] = ()
        else:
            report, exit_code, missing_variables = run_live_smoke_sync()
        sys.stdout.write(report.model_dump_json(indent=2) + "\n")
        if report.passed:
            sys.stderr.write(
                "LIVE_SMOKE PASS "
                f"chats={report.logical_chat_calls} attempts={report.provider_attempts} "
                f"cost_cny={report.known_cost_cny}\n"
            )
        else:
            missing_summary = f" missing={','.join(missing_variables)}" if missing_variables else ""
            sys.stderr.write(f"LIVE_SMOKE FAIL {report.error_category}{missing_summary}\n")
        return exit_code

    selected_case = parsed.case_id or environment_case
    report, exit_code = run_evaluation_sync(selected_case=selected_case)
    sys.stdout.write(report.model_dump_json(indent=2) + "\n")

    if report.error_category is not None:
        sys.stderr.write(f"EVAL CONFIGURATION_ERROR {report.error_category}\n")
        return exit_code
    for case in report.cases:
        status = "PASS" if case.passed else "FAIL"
        sys.stderr.write(f"[{status}] {case.case_id}\n")
    passed_cases = sum(case.passed for case in report.cases)
    overall_status = "PASS" if report.passed else "FAIL"
    sys.stderr.write(f"EVAL {overall_status} {passed_cases}/{len(report.cases)}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
