"""Validate real Make/pytest shard routing without running fixtures or test bodies."""

from __future__ import annotations

import os
import re
import selectors
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from .ci_contract import DEMO_NODE, INTEGRATION_SHARD_COUNT, RETRIEVAL_NODE
    from .ci_diagnostics import collection_diagnostic
else:
    from ci_contract import DEMO_NODE, INTEGRATION_SHARD_COUNT, RETRIEVAL_NODE
    from ci_diagnostics import collection_diagnostic

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_LIMIT = 2_000_000


class RoutingError(Exception):
    pass


def bounded_command(command, *, root, environment, timeout=60):
    """Drain both pipes with a combined byte cap and a wall-clock timeout."""
    try:
        with subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        ) as process:
            chunks = {"stdout": [], "stderr": []}
            size = 0
            deadline = time.monotonic() + timeout
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise RoutingError("category=subprocess_timeout")
                    for key, _ in selector.select(
                        timeout=min(0.1, max(0, deadline - time.monotonic()))
                    ):
                        data = os.read(key.fileobj.fileno(), 16384)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        size += len(data)
                        if size > OUTPUT_LIMIT:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            raise RoutingError("category=subprocess_output_limit")
                        chunks[key.data].append(data)
                try:
                    code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise RoutingError("category=subprocess_timeout") from None
        return code, (b"".join(chunks["stdout"]) + b"\n" + b"".join(chunks["stderr"])).decode(
            "utf-8", errors="replace"
        )
    except OSError:
        raise RoutingError("category=subprocess_start_failed") from None


def offline_environment():
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "PYTHONPATH"
        and not key.startswith(
            ("PYTEST_", "PF_", "OPENAI_", "DASHSCOPE_", "TAVILY_", "LANGFUSE_", "SUPABASE_")
        )
    }
    environment.update(
        PF_LLM_MODE="fake",
        PF_SEARCH_MODE="fake",
        PF_AUTH_MODE="fake",
        PF_TRACE_MODE="off",
        UV_OFFLINE="true",
    )
    return environment


def checked_command(command, *, phase, root, environment):
    try:
        code, output = bounded_command(command, root=root, environment=environment)
    except RoutingError as error:
        raise RoutingError(f"phase={phase} {error}") from None
    if code:
        raise RoutingError(
            f"phase={phase} exit_code={code} {collection_diagnostic(output, root=root)}"
        )
    return output


def collected_ids(output):
    ids = tuple(
        line
        for line in output.splitlines()
        if line.startswith("tests/integration/") and "::" in line
    )
    if not ids or len(ids) != len(set(ids)):
        raise RoutingError("category=empty_or_duplicate_collection")
    return ids


def verify_partition(original, shards):
    if (
        len(shards) != INTEGRATION_SHARD_COUNT
        or not original
        or len(original) != len(set(original))
    ):
        raise RoutingError("category=invalid_partition")
    seen = set()
    for index, shard in enumerate(shards):
        assigned = set(sorted(original)[index::INTEGRATION_SHARD_COUNT])
        expected = tuple(node for node in original if node in assigned)
        if not shard or tuple(shard) != expected or seen.intersection(shard):
            raise RoutingError("category=invalid_partition")
        seen.update(shard)
    if seen != set(original) or {RETRIEVAL_NODE, DEMO_NODE}.intersection(seen):
        raise RoutingError("category=invalid_partition")


def routing_environment(workflow, index, root):
    try:
        core = workflow.split("      - name: PostgreSQL integration core\n", 1)[1].split(
            "      - name:", 1
        )[0]
        pythonpath = re.search(r"(?m)^          PYTHONPATH: (.+)$", core)
        addopts = re.search(r"(?m)^          PYTEST_ADDOPTS: >-\n((?:            .+\n)+)", core)
        if pythonpath is None or pythonpath[1] != "${{ github.workspace }}" or addopts is None:
            raise RoutingError("category=workflow_environment_mismatch")
        if "run: make test-integration-core" not in core:
            raise RoutingError("category=workflow_environment_mismatch")
        options = " ".join(addopts[1].split()).replace("${{ matrix.shard }}", str(index))
        if (
            f"--ci-shard-count={INTEGRATION_SHARD_COUNT}" not in options
            or "-p tests.ci_sharding" not in options
        ):
            raise RoutingError("category=workflow_environment_mismatch")
        environment = offline_environment()
        environment.update(PYTHONPATH=str(root), PYTEST_ADDOPTS=options + " --collect-only -q")
        return environment
    except (IndexError, TypeError):
        raise RoutingError("category=workflow_environment_mismatch") from None


def validate_routing(root=ROOT):
    environment = offline_environment()
    # Same pinned interpreter and locked uv path as the parent Make invocation.
    uv = os.environ.get("PF_CI_UV", "uv")
    dry = checked_command(
        ("make", "--dry-run", "test-integration-core", f"UV={uv}"),
        phase="selectors",
        root=root,
        environment=environment,
    )
    invocations = [
        shlex.split(line)
        for line in dry.replace("\\\n", " ").splitlines()
        if "run --locked pytest tests/integration" in line
    ]
    if len(invocations) != 1:
        raise RoutingError("phase=selectors category=make_selector_mismatch")
    selectors_ = invocations[0][invocations[0].index("pytest") + 1 :]
    environment.update(PYTHONPATH=str(root), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    output = checked_command(
        (
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "pytest_asyncio.plugin",
            "-o",
            "addopts=",
            "--collect-only",
            "-q",
            *selectors_,
        ),
        phase="unsharded",
        root=root,
        environment=environment,
    )
    original = collected_ids(output)
    workflow = (root / ".github/workflows/ci.yml").read_text()
    shards = []
    for index in range(INTEGRATION_SHARD_COUNT):
        output = checked_command(
            ("make", "test-integration-core", f"UV={uv}"),
            phase=f"shard-{index}",
            root=root,
            environment=routing_environment(workflow, index, root),
        )
        shards.append(collected_ids(output))
    verify_partition(original, shards)
    return len(original), tuple(len(shard) for shard in shards)


def main():
    try:
        total, counts = validate_routing()
        message = (
            f"CI routing PASS: total={total} shards={counts}; collection only, no test execution."
        )
        code = 0
    except RoutingError as error:
        message, code = f"CI routing FAIL: {error}", 1
    except Exception:
        message, code = "CI routing FAIL: category=routing_configuration_error", 1
    print(message)
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        try:
            with Path(destination).open("a") as stream:
                stream.write(message + "\n")
        except OSError:
            return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
