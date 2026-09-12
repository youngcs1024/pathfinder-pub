"""Synthetic routing failures and bounded diagnostics; real collection lives in Make."""

import os
import sys
from pathlib import Path

import pytest

from scripts import ci_validate_routing as routing
from scripts.ci_diagnostics import collection_diagnostic

ROOT = Path(__file__).resolve().parents[2]
NODES = tuple(f"tests/integration/test_sample.py::test_{i}" for i in (3, 1, 2, 0))
SHARDS = tuple(tuple(n for n in NODES if n in sorted(NODES)[i::2]) for i in range(2))


def test_partition_preserves_original_execution_order_and_exact_coverage():
    routing.verify_partition(NODES, SHARDS)


@pytest.mark.parametrize(
    "shards",
    [
        ((), ()),
        (NODES,),
        (NODES, NODES),
        (SHARDS[1], SHARDS[0]),
        (SHARDS[0][:-1], SHARDS[1]),
        (tuple(reversed(SHARDS[0])), SHARDS[1]),
    ],
)
def test_partition_rejects_empty_overlap_missing_and_reordered_assignments(shards):
    with pytest.raises(routing.RoutingError, match="invalid_partition"):
        routing.verify_partition(NODES, shards)


def test_special_dedicated_tests_cannot_also_run_in_core():
    nodes = tuple(sorted((routing.DEMO_NODE, routing.RETRIEVAL_NODE)))
    with pytest.raises(routing.RoutingError, match="invalid_partition"):
        routing.verify_partition(nodes, ((nodes[0],), (nodes[1],)))


@pytest.mark.parametrize("output", ["", "CANARY", NODES[0] + "\n" + NODES[0]])
def test_invalid_collection_never_echoes_node_parameters(output):
    with pytest.raises(routing.RoutingError, match="empty_or_duplicate_collection") as error:
        routing.collected_ids(output)
    assert "CANARY" not in str(error.value)


def test_workflow_plugin_environment_matches_real_make_invocation(monkeypatch):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "CANARY")
    monkeypatch.setenv("PF_LLM_MODE", "qwen")
    environment = routing.routing_environment(
        (ROOT / ".github/workflows/ci.yml").read_text(), 1, ROOT
    )
    assert environment["PYTHONPATH"] == str(ROOT)
    assert "--ci-shard-index=1 --ci-shard-count=2" in environment["PYTEST_ADDOPTS"]
    assert "--collect-only -q" in environment["PYTEST_ADDOPTS"]
    assert "-p tests.ci_sharding" in environment["PYTEST_ADDOPTS"]
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in environment
    assert "DASHSCOPE_API_KEY" not in environment
    assert environment["PF_LLM_MODE"] == "fake"


@pytest.mark.parametrize(
    "old,new",
    [
        ("PYTHONPATH:", "REMOVED:"),
        ("--ci-shard-count=2", "--ci-shard-count=3"),
        ("-p tests.ci_sharding", "-p missing"),
    ],
)
def test_workflow_environment_drift_fails_closed(old, new):
    with pytest.raises(routing.RoutingError, match="workflow_environment_mismatch"):
        routing.routing_environment(
            (ROOT / ".github/workflows/ci.yml").read_text().replace(old, new), 0, ROOT
        )


def test_collection_diagnostics_allow_only_existing_repository_test_files(tmp_path):
    directory = tmp_path / "tests"
    directory.mkdir()
    (directory / "test_safe.py").write_text("")
    (directory / "escape.py").symlink_to(ROOT / "Makefile")
    diagnostic = collection_diagnostic(
        "ImportError: SECRET-CANARY\nERROR collecting tests/test_safe.py::test_x[PARAM-CANARY]\n"
        "ERROR collecting /private/CANARY.py\nERROR collecting tests/../CANARY.py\n"
        "ERROR collecting tests/escape.py\nERROR collecting tests/missing.py",
        root=tmp_path,
    )
    assert diagnostic == "category=import_error files=tests/test_safe.py"


def test_diagnostic_input_and_file_count_are_bounded(tmp_path):
    directory = tmp_path / "tests"
    directory.mkdir()
    for index in range(12):
        (directory / f"test_{index}.py").write_text("")
    output = "\n".join(f"ERROR collecting tests/test_{i}.py" for i in range(12))
    assert collection_diagnostic(output, root=tmp_path).count("tests/") == 8
    assert (
        collection_diagnostic("x" * 262144 + "ImportError", root=tmp_path)
        == "category=collection_error"
    )


def test_subprocess_failure_summary_never_copies_raw_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        routing, "bounded_command", lambda *a, **kw: (2, "ImportError: SECRET-CANARY")
    )

    def fail():
        routing.checked_command(("unused",), phase="shard-0", root=tmp_path, environment={})

    monkeypatch.setattr(routing, "validate_routing", fail)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert routing.main() == 1
    output = capsys.readouterr().out + summary.read_text()
    assert "phase=shard-0 exit_code=2 category=import_error" in output
    assert "CANARY" not in output


@pytest.mark.parametrize(
    "body,timeout,category",
    [
        ("import time; time.sleep(5)", 0.05, "subprocess_timeout"),
        ("print('x' * 4096)", 5, "subprocess_output_limit"),
    ],
)
def test_subprocess_has_time_and_output_bounds(tmp_path, monkeypatch, body, timeout, category):
    monkeypatch.setattr(routing, "OUTPUT_LIMIT", 1024)
    with pytest.raises(routing.RoutingError, match=category):
        routing.bounded_command(
            (sys.executable, "-c", body),
            root=tmp_path,
            environment=dict(os.environ),
            timeout=timeout,
        )


def test_subprocess_drains_stdout_and_stderr_and_propagates_exit_code(tmp_path):
    code, output = routing.bounded_command(
        (
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)",
        ),
        root=tmp_path,
        environment=dict(os.environ),
    )
    assert code == 3 and "out" in output and "err" in output


def test_make_routing_entry_uses_locked_toolchain():
    makefile = (ROOT / "Makefile").read_text()
    assert (
        'test-ci-routing: verify-toolchain\n\tPF_CI_UV="$(UV)" '
        "$(UV) run --locked python scripts/ci_validate_routing.py" in makefile
    )


def test_routing_orchestrates_real_make_selectors_and_both_workflow_shards(monkeypatch):
    calls = []

    def command(args, *, phase, root, environment):
        calls.append((args, phase, environment))
        if phase == "selectors":
            return "uv run --locked pytest tests/integration --deselect " + routing.DEMO_NODE
        nodes = NODES if phase == "unsharded" else SHARDS[int(phase[-1])]
        return "\n".join(nodes)

    monkeypatch.setattr(routing, "checked_command", command)
    assert routing.validate_routing() == (4, (2, 2))
    assert [phase for _, phase, _ in calls] == ["selectors", "unsharded", "shard-0", "shard-1"]
    assert calls[0][0][:3] == ("make", "--dry-run", "test-integration-core")
    assert "--deselect" in calls[1][0] and routing.DEMO_NODE in calls[1][0]
    for index, (args, _, environment) in enumerate(calls[2:]):
        assert args[:2] == ("make", "test-integration-core")
        assert f"--ci-shard-index={index}" in environment["PYTEST_ADDOPTS"]
        assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in environment


def test_missing_make_selector_fails_before_collection(monkeypatch):
    monkeypatch.setattr(routing, "checked_command", lambda *a, **kw: "unexpected CANARY")
    with pytest.raises(
        routing.RoutingError, match="phase=selectors category=make_selector_mismatch"
    ):
        routing.validate_routing()
