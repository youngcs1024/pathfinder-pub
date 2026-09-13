from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(
    *command: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("IMAGE_TAG", None)
    env.pop("PATHFINDER_IMAGE", None)
    if env_overrides is not None:
        env.update(env_overrides)
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


@pytest.mark.parametrize(
    ("target", "expected_command"),
    [
        (
            "up",
            "PF_PUBLIC_DOMAIN=localhost docker compose -f compose.yaml -f compose.dev.yaml "
            "up --detach --wait --wait-timeout 60 postgres",
        ),
        (
            "down",
            "PF_PUBLIC_DOMAIN=localhost docker compose -f compose.yaml -f compose.dev.yaml "
            "down --timeout 10",
        ),
        ("migrate", "uv run --locked alembic upgrade head"),
        ("run-api", "uv run --locked python -m app.main"),
        ("run-worker", "uv run --locked python -m app.worker.main"),
        (
            "test-unit",
            "uv run --locked pytest tests/unit",
        ),
        (
            "test-architecture",
            "uv run --locked pytest tests/architecture",
        ),
        (
            "test-operational-contracts",
            "uv run --locked pytest tests/operational",
        ),
        (
            "test-eval-contracts",
            "uv run --locked pytest tests/evals",
        ),
        (
            "test-integration-core",
            "uv run --locked pytest tests/integration --deselect "
            "tests/integration/db/test_retrieval_benchmark.py::"
            "test_real_db_benchmark_pipeline_filters_accounting_and_determinism --deselect "
            "tests/integration/db/test_gate8_demo.py::"
            "test_gate8_demo_complete_application_flow",
        ),
        (
            "test-integration",
            "uv run --locked pytest tests/integration",
        ),
        (
            "audit-deps",
            "uv --preview-features audit-command audit --frozen",
        ),
        ("evals", "uv run --locked python -m tests.evals"),
        (
            "demo",
            "uv run --locked pytest -q "
            "tests/integration/db/test_gate8_demo.py::test_gate8_demo_complete_application_flow",
        ),
        ("live-smoke", "uv run --locked python -m tests.evals --live-smoke"),
        (
            "live-evals-chat-accepted",
            "uv run --locked python -m tests.evals --live-chat-accepted",
        ),
        (
            "live-evals-retrieval",
            "run --locked pytest -q -s "
            "tests/integration/db/test_live_retrieval.py::"
            "test_real_qwen_embedding_retrieval_benchmark",
        ),
        (
            "live-evals-retrieval-accepted",
            "run --locked pytest -q -s "
            "tests/integration/db/test_live_retrieval.py::"
            "test_accepted_qwen_embedding_retrieval_benchmark",
        ),
        ("image-build", 'docker build --tag "pathfinder-ci:local" .'),
        (
            "image-smoke",
            'PF_PUBLIC_DOMAIN=localhost PATHFINDER_IMAGE="pathfinder-ci:local" '
            "docker compose run --rm --no-deps "
            "--entrypoint python api",
        ),
    ],
)
def test_make_targets_expand_to_real_commands(target: str, expected_command: str) -> None:
    result = _run("make", "--dry-run", target)
    expanded = result.stdout.replace("\\\n\t", "")

    assert result.returncode == 0, result.stderr
    assert expected_command in expanded


def test_live_retrieval_target_sets_explicit_opt_in_and_isolated_test_selector() -> None:
    result = _run("make", "--dry-run", "live-evals-retrieval")

    assert result.returncode == 0, result.stderr
    assert "PF_RUN_GATE11_LIVE_RETRIEVAL=1" in result.stdout
    assert 'TMPDIR="/tmp"' in result.stdout
    assert (
        "tests/integration/db/test_live_retrieval.py::test_real_qwen_embedding_retrieval_benchmark"
    ) in result.stdout


def test_accepted_live_targets_are_explicit_and_baseline_accept_requires_inputs() -> None:
    retrieval = _run(
        "make",
        "--dry-run",
        "live-evals-retrieval-accepted",
        "REPORT=/tmp/retrieval.json",
    )
    assert retrieval.returncode == 0, retrieval.stderr
    assert "PF_RUN_GATE11_LIVE_RETRIEVAL_ACCEPTED=1" in retrieval.stdout
    assert 'PF_GATE11_ACCEPTED_RETRIEVAL_REPORT="/tmp/retrieval.json"' in retrieval.stdout
    assert "test_accepted_qwen_embedding_retrieval_benchmark" in retrieval.stdout

    acceptance = _run(
        "make",
        "--dry-run",
        "live-evals-baseline-accept",
        "CHAT_REPORT=chat.json",
        "RETRIEVAL_REPORT=retrieval.json",
        "REASON=reviewed",
    )
    assert acceptance.returncode == 0, acceptance.stderr
    assert "python -m tests.evals.live_baseline" in acceptance.stdout
    assert '--chat-report "chat.json"' in acceptance.stdout
    assert '--retrieval-report "retrieval.json"' in acceptance.stdout
    assert '--reason "reviewed"' in acceptance.stdout


@pytest.mark.parametrize(
    ("target", "expected_command"),
    [
        ("image-build", 'docker build --tag "pathfinder-ci:test-sha" .'),
        (
            "image-smoke",
            'PF_PUBLIC_DOMAIN=localhost PATHFINDER_IMAGE="pathfinder-ci:test-sha" '
            "docker compose run --rm --no-deps "
            "--entrypoint python api",
        ),
    ],
)
def test_image_targets_honor_explicit_image_tag_override(
    target: str,
    expected_command: str,
) -> None:
    result = _run(
        "make",
        "--dry-run",
        target,
        "IMAGE_TAG=pathfinder-ci:test-sha",
    )

    assert result.returncode == 0, result.stderr
    assert expected_command in result.stdout


@pytest.mark.parametrize(
    ("target", "override"),
    [
        ("up", "COMPOSE=/bin/false"),
        ("down", "COMPOSE=/bin/false"),
        ("migrate", "UV=/bin/false"),
        ("run-api", "UV=/bin/false"),
        ("run-worker", "UV=/bin/false"),
        ("test-unit", "UV=/bin/false"),
        ("test-architecture", "UV=/bin/false"),
        ("test-operational-contracts", "UV=/bin/false"),
        ("test-eval-contracts", "UV=/bin/false"),
        ("test-integration-core", "UV=/bin/false"),
        ("test-integration", "UV=/bin/false"),
        ("audit-deps", "UV=/bin/false"),
        ("evals", "UV=/bin/false"),
        ("demo", "UV=/bin/false"),
        ("live-smoke", "UV=/bin/false"),
        ("live-evals-chat-accepted", "UV=/bin/false"),
        ("live-evals-retrieval", "UV=/bin/false"),
        ("live-evals-retrieval-accepted", "UV=/bin/false"),
        ("live-evals-baseline-accept", "UV=/bin/false"),
        ("image-build", "DOCKER=/bin/false"),
        ("image-smoke", "COMPOSE=/bin/false"),
    ],
)
def test_make_targets_propagate_command_failure(target: str, override: str) -> None:
    result = _run("make", target, override)

    assert result.returncode != 0


def test_demo_forces_offline_modes_and_passes_test_only_failure_selector() -> None:
    result = _run("make", "--dry-run", "demo", "DEMO_FAILURE=rag")

    assert result.returncode == 0, result.stderr
    assert "PF_LLM_MODE=fake" in result.stdout
    assert "PF_SEARCH_MODE=fake" in result.stdout
    assert "PF_AUTH_MODE=fake" in result.stdout
    assert "PF_TRACE_MODE=off" in result.stdout
    assert 'PF_GATE87_DEMO_FAILURE="rag"' in result.stdout


def test_benchmark_help_requires_no_toolchain_or_docker():
    result = _run(
        "make", "benchmark-help", "UV=/bin/false", "DOCKER=/bin/false", "COMPOSE=/bin/false"
    )
    assert result.returncode == 0
    for target in ("smoke", "capacity", "queue", "retrieval", "faults"):
        assert f"make benchmark-{target}" in result.stdout
    assert "fake/fake/fake/off" in result.stdout
    assert "128 HTTP requests / 64 calls / 40 business seconds" in result.stdout
    assert "GNU Make returns 2" in result.stdout
    assert "create-only" in result.stdout


@pytest.mark.parametrize(
    "target,command,prefix,profile,authorization",
    [
        ("smoke", "smoke", "SMOKE", "instant-v1", "e510_local_smoke_user_approved_v1"),
        (
            "capacity",
            "capacity",
            "CAPACITY",
            "capacity-e56-v1",
            "e56_local_capacity_user_approved_v1",
        ),
        ("queue", "queue", "QUEUE", "queue-e57-v1", "e57_local_queue_user_approved_v1"),
        (
            "retrieval",
            "retrieval-scale",
            "RETRIEVAL",
            "retrieval-e58-v1",
            "e58_local_retrieval_user_approved_v1",
        ),
        ("faults", "faults", "FAULTS", "faults-e59-v1", "e59_local_faults_user_approved_v1"),
    ],
)
@pytest.mark.parametrize("business_exit", [0, 1, 130])
def test_benchmark_make_preserves_failure_without_confusing_business_exit(
    tmp_path, target, command, prefix, profile, authorization, business_exit
):
    uv = tmp_path / "uv-stub"
    calls = tmp_path / "calls.jsonl"
    uv.write_text(
        f"#!{sys.executable}\n"
        + dedent("""\
        import json
        import os
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        if args == ["--version"]:
            print("uv 0.11.32")
        elif args == ["lock", "--check"]:
            pass
        elif args[:4] == ["run", "--locked", "python", "-c"]:
            print("CPython" if "python_implementation" in args[-1] else "3.12.13")
        elif args[:5] == ["run", "--locked", "python", "-m", "tests.performance"]:
            with Path(os.environ["E510_TEST_CALLS"]).open("a") as stream:
                stream.write(json.dumps(args) + "\\n")
            sys.exit(int(os.environ["E510_TEST_EXIT"]))
        else:
            sys.exit(99)
    """)
    )
    uv.chmod(0o700)
    output = tmp_path / "private output"
    environment = dict(os.environ, E510_TEST_CALLS=str(calls), E510_TEST_EXIT=str(business_exit))
    args = ["run", "--locked", "python", "-m", "tests.performance", command, "--profile", profile]
    if target == "smoke":
        args += ["--mode", "application"]
    args += ["--authorization", authorization, "--output", str(output)]
    direct = subprocess.run([str(uv), *args], env=environment, capture_output=True, timeout=10)
    assert direct.returncode == business_exit
    result = _run(
        "make",
        "--no-print-directory",
        f"benchmark-{target}",
        f"UV={uv}",
        f"{prefix}_OUTPUT={output}",
        f"{prefix}_AUTHORIZATION={authorization}",
        env_overrides=environment,
    )
    assert result.returncode == (0 if business_exit == 0 else 2)
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [args, args]
    assert not output.exists()
