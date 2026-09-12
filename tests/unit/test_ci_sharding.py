from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.ci_sharding import select_node_ids

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ("-p", "tests.ci_sharding")


def _offline_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "PYTHONPATH"
        and not key.startswith(
            (
                "PYTEST_",
                "PF_",
                "OPENAI_",
                "DASHSCOPE_",
                "TAVILY_",
                "LANGFUSE_",
                "SUPABASE_",
            )
        )
    }
    environment.update(
        PF_LLM_MODE="fake",
        PF_SEARCH_MODE="fake",
        PF_AUTH_MODE="fake",
        PF_TRACE_MODE="off",
        PF_RUN_GATE11_LIVE_RETRIEVAL="0",
        PF_RUN_GATE11_LIVE_RETRIEVAL_ACCEPTED="0",
    )
    return environment


def _pytest(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = _offline_environment()
    environment.update(
        PYTHONPATH=str(PROJECT_ROOT),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    return subprocess.run(
        (
            str(Path(sys.executable).with_name("pytest")),
            "-p",
            "no:cacheprovider",
            "-p",
            "pytest_asyncio.plugin",
            "-o",
            "addopts=",
            *args,
        ),
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _collected(directory: Path, *args: str) -> tuple[str, ...]:
    result = _pytest(directory, "--collect-only", "-q", *args)
    if result.returncode:
        pytest.fail(f"pytest collection failed: exit_code={result.returncode}; raw output withheld")
    return tuple(
        line
        for line in result.stdout.splitlines()
        if line.startswith(("test_cases.py::", "tests/integration/")) and "::" in line
    )


@pytest.fixture
def sample_suite(tmp_path: Path) -> Path:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_cases.py").write_text(
        "import pytest\n"
        "def test_z(): pass\n"
        "@pytest.mark.parametrize('value', [0, 1, 2])\n"
        "def test_a(value): pass\n"
        "def test_excluded(): raise AssertionError('must be deselected')\n"
        "def test_new(): pass\n",
        encoding="utf-8",
    )
    return tmp_path


def test_partition_is_order_independent_disjoint_complete_and_includes_new_ids() -> None:
    node_ids = ("test_z", "test_a[1]", "test_new", "test_a[0]", "test_a[2]")
    shards = [select_node_ids(node_ids, index=index, count=2) for index in range(2)]
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == set(node_ids)
    assert abs(len(shards[0]) - len(shards[1])) <= 1
    for index in range(2):
        assert select_node_ids(tuple(reversed(node_ids)), index=index, count=2) == shards[index]


def test_duplicate_node_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="unique node IDs"):
        select_node_ids(("test_a", "test_a"), index=0, count=2)


@pytest.mark.parametrize(
    "options",
    [
        (),
        ("--ci-shard-index=0",),
        ("--ci-shard-count=2",),
        ("--ci-shard-index=bad", "--ci-shard-count=2"),
        ("--ci-shard-index=0", "--ci-shard-count=bad"),
        ("--ci-shard-index=-1", "--ci-shard-count=2"),
        ("--ci-shard-index=0", "--ci-shard-count=0"),
        ("--ci-shard-index=0", "--ci-shard-count=-1"),
        ("--ci-shard-index=2", "--ci-shard-count=2"),
    ],
)
def test_plugin_rejects_missing_or_invalid_options(
    sample_suite: Path, options: tuple[str, ...]
) -> None:
    result = _pytest(sample_suite, "--collect-only", *PLUGIN, *options)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR


def test_empty_shard_fails_instead_of_succeeding(sample_suite: Path) -> None:
    result = _pytest(
        sample_suite,
        "--collect-only",
        "test_cases.py::test_z",
        *PLUGIN,
        "--ci-shard-index=1",
        "--ci-shard-count=2",
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "CI shard selected no tests" in result.stderr


def test_native_deselection_precedes_partition_and_original_order_is_preserved(
    sample_suite: Path,
) -> None:
    selectors = ("--deselect=test_cases.py::test_excluded",)
    original = _collected(sample_suite, *selectors)
    assert len(original) == 5
    assert original[0] == "test_cases.py::test_z"
    shards = [
        _collected(
            sample_suite,
            *selectors,
            *PLUGIN,
            f"--ci-shard-index={index}",
            "--ci-shard-count=2",
        )
        for index in range(2)
    ]
    assert set(shards[0]).isdisjoint(shards[1])
    assert set(shards[0]) | set(shards[1]) == set(original)
    for index, shard in enumerate(shards):
        assigned = set(sorted(original)[index::2])
        assert shard == tuple(node_id for node_id in original if node_id in assigned)
    # Explicit loading in earlier child processes must not affect an ordinary run.
    result = _pytest(sample_suite, "-q", *selectors)
    assert result.returncode == 0
    assert "5 passed" in result.stdout


def test_failure_in_selected_shard_is_not_hidden(sample_suite: Path) -> None:
    result = _pytest(
        sample_suite,
        "-q",
        "test_cases.py::test_excluded",
        *PLUGIN,
        "--ci-shard-index=0",
        "--ci-shard-count=1",
    )
    assert result.returncode == pytest.ExitCode.TESTS_FAILED


def test_import_failure_preserves_original_error_instead_of_empty_shard(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_broken.py").write_text("raise ImportError('synthetic collection failure')\n")
    result = _pytest(
        tmp_path, "--collect-only", "-q", *PLUGIN, "--ci-shard-index=0", "--ci-shard-count=2"
    )
    assert result.returncode == 2
    assert "ImportError" in result.stdout
    assert "CI shard selected no tests" not in result.stdout + result.stderr
