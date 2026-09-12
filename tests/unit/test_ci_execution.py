"""Exercise local CI entrypoints with deterministic subprocesses, never providers or Docker."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(("format_status", "check_status"), [(0, 0), (1, 0), (0, 2), (1, 2)])
@pytest.mark.parametrize("lock_status", [0, 1])
@pytest.mark.parametrize("target", ["lint", "lint-format", "lint-rules"])
def test_make_lint_collects_both_checks_without_hiding_failure(
    tmp_path, format_status, check_status, lock_status, target
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
            with Path(os.environ["CI_TEST_UV_CALLS"]).open("a") as log:
                log.write(json.dumps(args) + "\\n")
            if args == ["--version"]:
                print("uv 0.11.32")
            elif args == ["lock", "--check"]:
                sys.exit(int(os.environ["CI_TEST_LOCK_STATUS"]))
            elif args[:3] == ["run", "--locked", "python"]:
                print("CPython" if "python_implementation" in args[-1] else "3.12.13")
            elif args == ["run", "--locked", "ruff", "format", "--check", "."]:
                sys.exit(int(os.environ["CI_TEST_FORMAT_STATUS"]))
            elif args == ["run", "--locked", "ruff", "check", "."]:
                sys.exit(int(os.environ["CI_TEST_CHECK_STATUS"]))
            else:
                sys.exit(99)
        """),
        encoding="utf-8",
    )
    uv.chmod(0o700)
    env = dict(os.environ)
    env.update(
        CI_TEST_UV_CALLS=str(calls),
        CI_TEST_FORMAT_STATUS=str(format_status),
        CI_TEST_CHECK_STATUS=str(check_status),
        CI_TEST_LOCK_STATUS=str(lock_status),
    )
    result = subprocess.run(
        ["make", "--no-print-directory", "-f", str(PROJECT_ROOT / "Makefile"), target, f"UV={uv}"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    ruff_commands = [args for args in commands if args[:3] == ["run", "--locked", "ruff"]]
    expected = []
    statuses = [lock_status]
    if target in {"lint", "lint-format"}:
        expected.append(["run", "--locked", "ruff", "format", "--check", "."])
        statuses.append(format_status)
    if target in {"lint", "lint-rules"}:
        expected.append(["run", "--locked", "ruff", "check", "."])
        statuses.append(check_status)
    assert ruff_commands == ([] if lock_status else expected)
    assert (result.returncode == 0) is (not any(statuses))


def test_pytest_configuration_collects_same_basenames_and_repository_script_imports(tmp_path):
    for directory in ("first", "second"):
        root = tmp_path / directory
        root.mkdir()
        (root / "test_same_name.py").write_text(
            "from scripts.ci_gate import HEAVY_JOBS\n"
            "def test_script_import():\n"
            "    assert 'quality' in HEAVY_JOBS\n",
            encoding="utf-8",
        )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("pytest")),
            "-p",
            "pytest_asyncio.plugin",
            "-c",
            str(PROJECT_ROOT / "pyproject.toml"),
            str(tmp_path / "first"),
            str(tmp_path / "second"),
            "-q",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


@pytest.mark.parametrize("scenario", ["valid", "import_error", "empty", "unowned"])
def test_make_collect_uses_real_pytest_without_executing_fixtures(tmp_path, scenario):
    suite_root = tmp_path / "tests"
    suite = suite_root / ("unowned" if scenario == "unowned" else "unit")
    suite.mkdir(parents=True)
    (suite_root / "__init__.py").write_text("")
    (suite_root / "ci_coverage.py").write_bytes(
        (PROJECT_ROOT / "tests/ci_coverage.py").read_bytes()
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "__init__.py").write_text("")
    (scripts / "ci_contract.py").write_bytes((PROJECT_ROOT / "scripts/ci_contract.py").read_bytes())
    (scripts / "ci_diagnostics.py").write_bytes(
        (PROJECT_ROOT / "scripts/ci_diagnostics.py").read_bytes()
    )
    if scenario != "empty":
        (suite / "test_collection.py").write_text(
            "raise ImportError('SECRET-CANARY')\n"
            if scenario == "import_error"
            else "import pytest\n"
            "@pytest.fixture(autouse=True)\n"
            "def no_fixture_execution():\n"
            "    raise AssertionError('fixture must not execute')\n"
            "def test_only_collect():\n"
            "    raise AssertionError('body must not execute')\n",
            encoding="utf-8",
        )
    uv = tmp_path / "uv-stub"
    uv.write_text(
        f"#!{sys.executable}\n"
        + dedent("""\
            import os
            import sys
            args = sys.argv[1:]
            if args == ["--version"]:
                print("uv 0.11.32")
            elif args == ["lock", "--check"]:
                pass
            elif args[:3] == ["run", "--locked", "python"]:
                print("CPython" if "python_implementation" in args[-1] else "3.12.13")
            elif args == [
                "run", "--locked", "pytest", "-p", "tests.ci_coverage",
                "tests", "--collect-only", "-q",
            ]:
                os.execv(sys.executable, [sys.executable, "-m", "pytest", *args[3:]])
            else:
                sys.exit(99)
        """),
        encoding="utf-8",
    )
    uv.chmod(0o700)
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    summary = tmp_path / "summary.md"
    environment["GITHUB_STEP_SUMMARY"] = str(summary)
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-f",
            str(PROJECT_ROOT / "Makefile"),
            "test-collect",
            f"UV={uv}",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert (result.returncode == 0) is (scenario == "valid")
    if scenario == "valid":
        assert "1 test collected" in result.stdout
    elif scenario == "import_error":
        assert "ERROR collecting" in result.stdout
        diagnostic = summary.read_text()
        assert (
            "phase=collect exit_code=2 category=import_error files=tests/unit/test_collection.py"
            in diagnostic
        )
        assert "CANARY" not in diagnostic
    elif scenario == "empty":
        assert "empty or duplicate CI collection" in result.stderr
    else:
        assert "exactly one execution target" in result.stderr


@pytest.mark.parametrize(("version", "exit_code"), [("24.20.0", 0), ("24.20.0", 1), ("22.0.0", 0)])
def test_ui_make_target_checks_runtime_and_propagates_test_failure(tmp_path, version, exit_code):
    binary = tmp_path / "node"
    calls_file = tmp_path / "calls"
    binary.write_text(
        f"#!{sys.executable}\n"
        + dedent("""\
            import os
            import sys
            from pathlib import Path

            if sys.argv[1:] == ["--version"]:
                print("v" + os.environ["CI_UI_NODE_VERSION"])
            else:
                Path(os.environ["CI_UI_NODE_CALLS"]).write_text(" ".join(sys.argv[1:]))
                sys.exit(int(os.environ["CI_UI_NODE_EXIT"]))
        """),
        encoding="utf-8",
    )
    binary.chmod(0o700)
    env = dict(os.environ)
    env.update(
        PATH=f"{tmp_path}:{env['PATH']}",
        CI_UI_NODE_VERSION=version,
        CI_UI_NODE_EXIT=str(exit_code),
        CI_UI_NODE_CALLS=str(calls_file),
    )
    result = subprocess.run(
        ["make", "--no-print-directory", "test-ui"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert (result.returncode == 0) is (version == "24.20.0" and exit_code == 0)
    if version == "24.20.0":
        command = calls_file.read_text()
        assert command.startswith("--test ")
        assert "tests/ui/pathfinder.test.cjs" in command
    else:
        assert not calls_file.exists()
