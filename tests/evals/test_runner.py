from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tests.evals.harness as harness
from app.agents.research_graph import ResearchGraphProtocolError
from app.llm.ports import ChatModelResult, ModelUsage
from tests.evals.contracts import EvalReportV3
from tests.evals.harness import DEFAULT_DATASET_PATH, run_evaluation
from tests.evals.live_contracts import LiveSmokeReportV1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_ENVIRONMENT_NAMES = {
    "PF_EVAL_CASE",
    "PF_LLM_MODE",
    "PF_QWEN_WORKSPACE_ID",
    "DASHSCOPE_API_KEY",
    "PF_SEARCH_MODE",
    "TAVILY_API_KEY",
    "PF_TRACE_MODE",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_SAMPLE_RATE",
}


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        (sys.executable, "-m", "tests.evals", *arguments),
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_live_cli(
    environment_updates: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value for key, value in os.environ.items() if key not in LIVE_ENVIRONMENT_NAMES
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.update(environment_updates or {})
    return subprocess.run(
        (sys.executable, "-m", "tests.evals", "--live-smoke"),
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


async def test_full_and_single_case_runner_return_strict_reports() -> None:
    full_report, full_exit = await run_evaluation()
    single_report, single_exit = await run_evaluation(selected_case="normal_application")

    assert full_exit == 0
    assert full_report.passed is True
    assert len(full_report.cases) == 14
    assert full_report.schema_version == 3
    assert full_report.dataset_version == "research-v3"
    assert full_report.cost_available is False
    assert full_report.estimated_cost_cny is None
    assert full_report.input_tokens == sum(case.metrics.input_tokens for case in full_report.cases)
    assert full_report.output_tokens == sum(
        case.metrics.output_tokens for case in full_report.cases
    )
    assert single_exit == 0
    assert tuple(case.case_id for case in single_report.cases) == ("normal_application",)
    assert (
        EvalReportV3.model_validate_json(full_report.model_dump_json(), strict=True) == full_report
    )


def test_cli_stdout_is_byte_stable_machine_json_and_stderr_is_human_summary() -> None:
    first = _run_cli("--case", "normal_application")
    second = _run_cli("--case", "normal_application")

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == second.stdout
    report = EvalReportV3.model_validate_json(first.stdout, strict=True)
    assert report.passed is True
    assert report.selected_case == "normal_application"
    assert first.stderr == "[PASS] normal_application\nEVAL PASS 1/1\n"


def test_reserved_context_security_case_cli_is_deterministic() -> None:
    first = _run_cli("--case", "reserved_tool_context_injection")
    second = _run_cli("--case", "reserved_tool_context_injection")

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert (
        first.stderr == second.stderr == ("[PASS] reserved_tool_context_injection\nEVAL PASS 1/1\n")
    )


def test_full_cli_stdout_is_byte_stable() -> None:
    first = _run_cli()
    second = _run_cli()

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == second.stdout
    assert EvalReportV3.model_validate_json(first.stdout, strict=True).passed is True


def test_unknown_case_returns_exit_two_with_a_stable_configuration_report() -> None:
    result = _run_cli("--case", "does_not_exist")

    assert result.returncode == 2
    report = EvalReportV3.model_validate_json(result.stdout, strict=True)
    assert report.passed is False
    assert report.cases == ()
    assert report.artifact_identity is None
    assert report.comparison_metadata is None
    assert report.grader_aggregates == ()
    assert report.error_category == "unknown_case"
    assert result.stderr == "EVAL CONFIGURATION_ERROR unknown_case\n"


async def test_quality_threshold_failure_returns_exit_one(tmp_path: Path) -> None:
    lines = DEFAULT_DATASET_PATH.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["expectations"]["minimum_cited_sources"] = 3
    dataset_path = tmp_path / "quality_failure.jsonl"
    dataset_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    report, exit_code = await run_evaluation(dataset_path=dataset_path)

    assert exit_code == 1
    assert report.passed is False
    assert report.error_category is None
    assert report.artifact_identity is not None
    assert report.comparison_metadata is not None
    assert report.grader_aggregates
    source_grader = next(
        grader for grader in report.cases[0].graders if grader.name == "source_diversity"
    )
    assert source_grader.passed is False


@pytest.mark.parametrize(
    ("writer_content", "cause_category"),
    [
        ("{", "invalid_model_json"),
        ('{"summary":"not-a-list"}', "invalid_model_schema"),
    ],
)
async def test_malformed_writer_output_is_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
    writer_content: str,
    cause_category: str,
) -> None:
    malformed = ChatModelResult(
        content=writer_content,
        usage=ModelUsage(input_tokens=1, output_tokens=1),
    )
    monkeypatch.setattr(harness, "_writer_script", lambda *args: (malformed, malformed))

    report, exit_code = await run_evaluation(selected_case="normal_application")

    assert exit_code == 1
    assert report.cases[0].error_category == "invalid_output"
    assert (
        harness._protocol_error_category(
            ResearchGraphProtocolError(
                category="node_execution_failed",
                node_name="write_report",
                cause_category=cause_category,
            )
        )
        == "invalid_output"
    )


@pytest.mark.parametrize("cause_category", ["provider_timeout", "provider_unavailable"])
async def test_provider_protocol_failures_are_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
    cause_category: str,
) -> None:
    async def fail(_: object) -> None:
        raise ResearchGraphProtocolError(
            category="node_execution_failed",
            node_name="write_report",
            cause_category=cause_category,
        )

    monkeypatch.setattr(harness, "execute_eval_case", fail)
    report, exit_code = await run_evaluation(selected_case="normal_application")

    assert exit_code == 1
    assert report.cases[0].error_category == "provider_failure"


async def test_unexpected_graph_exception_is_graph_execution_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(_: object) -> None:
        raise RuntimeError("unexpected graph failure canary")

    monkeypatch.setattr(harness, "execute_eval_case", fail)
    report, exit_code = await run_evaluation(selected_case="normal_application")

    assert exit_code == 1
    assert report.cases[0].error_category == "graph_execution_failed"


async def test_invalid_final_graph_output_is_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidOutputGraph:
        async def ainvoke(self, *args: object, **kwargs: object) -> dict[str, object]:
            return {"output": {"schema_version": 2, "evidence_sufficient": True}}

    monkeypatch.setattr(harness, "build_research_state_graph", lambda *args: InvalidOutputGraph())
    report, exit_code = await run_evaluation(selected_case="normal_application")

    assert exit_code == 1
    assert report.cases[0].error_category == "invalid_output"


def test_live_smoke_missing_configuration_fails_before_network_and_lists_names_only() -> None:
    result = _run_live_cli()

    assert result.returncode == 2
    report = LiveSmokeReportV1.model_validate_json(result.stdout, strict=True)
    assert report.error_category == "invalid_settings"
    assert "DASHSCOPE_API_KEY" in result.stderr
    assert "TAVILY_API_KEY" in result.stderr


def test_live_smoke_rejects_wrong_modes_and_redacts_config_values_before_network() -> None:
    secret_canary = "live-smoke-secret-body-canary"
    result = _run_live_cli(
        {
            "PF_LLM_MODE": "fake",
            "PF_QWEN_WORKSPACE_ID": "workspace-test",
            "DASHSCOPE_API_KEY": secret_canary,
            "PF_SEARCH_MODE": "tavily",
            "TAVILY_API_KEY": secret_canary,
            "PF_TRACE_MODE": "langfuse",
            "LANGFUSE_PUBLIC_KEY": secret_canary,
            "LANGFUSE_SECRET_KEY": secret_canary,
            "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
            "LANGFUSE_SAMPLE_RATE": "1.0",
        }
    )

    assert result.returncode == 2
    report = LiveSmokeReportV1.model_validate_json(result.stdout, strict=True)
    assert report.error_category == "invalid_modes"
    assert secret_canary not in result.stdout
    assert secret_canary not in result.stderr


def test_live_smoke_requires_full_langfuse_sampling_before_network() -> None:
    result = _run_live_cli(
        {
            "PF_LLM_MODE": "qwen",
            "PF_QWEN_WORKSPACE_ID": "workspace-test",
            "DASHSCOPE_API_KEY": "qwen-test",
            "PF_SEARCH_MODE": "tavily",
            "TAVILY_API_KEY": "tavily-test",
            "PF_TRACE_MODE": "langfuse",
            "LANGFUSE_PUBLIC_KEY": "public-test",
            "LANGFUSE_SECRET_KEY": "secret-test",
            "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
            "LANGFUSE_SAMPLE_RATE": "0.5",
        }
    )

    assert result.returncode == 2
    report = LiveSmokeReportV1.model_validate_json(result.stdout, strict=True)
    assert report.error_category == "invalid_modes"


def test_live_smoke_and_environment_case_are_mutually_exclusive() -> None:
    result = _run_live_cli({"PF_EVAL_CASE": "normal_application"})

    assert result.returncode == 2
    report = LiveSmokeReportV1.model_validate_json(result.stdout, strict=True)
    assert report.error_category == "conflicting_arguments"


def test_live_smoke_and_case_flags_are_mutually_exclusive() -> None:
    result = _run_cli("--case", "normal_application", "--live-smoke")

    assert result.returncode == 2
    assert result.stdout == ""
    assert "not allowed with argument" in result.stderr


def test_live_chat_cli_missing_settings_is_sanitized_and_nonzero(monkeypatch):
    for key in LIVE_ENVIRONMENT_NAMES:
        monkeypatch.delenv(key, raising=False)
    result = _run_cli("--live-chat")
    assert result.returncode == 2
    assert result.stderr == "LIVE_CHAT FAIL invalid_settings\n"
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 5 and payload["suite_version"] == "live-eval-v5"
    assert payload["exploratory"] and not payload["complete"]
    assert payload["aggregate"]["provider_attempts"] == 0


@pytest.mark.parametrize("other", ["--live-smoke", "--case"])
def test_live_chat_cli_rejects_conflicting_modes(other):
    arguments = ["--live-chat", other]
    if other == "--case":
        arguments.append("normal_application")
    result = _run_cli(*arguments)
    assert result.returncode == 2
    assert not result.stdout


def test_live_chat_cli_rejects_environment_case(monkeypatch):
    monkeypatch.setenv("PF_EVAL_CASE", "normal_application")
    result = _run_cli("--live-chat")
    assert result.returncode == 2
    assert result.stderr == "LIVE_CHAT FAIL conflicting_arguments\n"
