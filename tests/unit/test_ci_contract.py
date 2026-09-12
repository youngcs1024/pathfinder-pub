"""Cross-check the shared contract against executable workflow and Make selectors."""

import re
from pathlib import Path

import pytest

from scripts.ci_contract import (
    CHECKS,
    DEMO_NODE,
    FULL_JOB_REQUIRED_STEPS,
    HEAVY_JOBS,
    JOB_PREREQUISITES,
    PROFILES,
    RETRIEVAL_NODE,
    SUMMARY_NAME,
    TEST_ROUTES,
    collection_ownership,
)
from scripts.ci_contract import TestRoute as Route
from tests.unit.test_ci_workflow import _jobs, _workflow

ROOT = Path(__file__).resolve().parents[2]


def assert_workflow_contract(workflow):
    jobs = _jobs(workflow)
    assert set(jobs) == set(CHECKS)
    for job, checks in CHECKS.items():
        header = jobs[job].split("    steps:", 1)[0]
        match = re.search(r"(?m)^    needs:([^\n]*)(?:\n((?:      - [a-z-]+\n)+))?", header)
        dependencies = (
            tuple(re.findall(r"[a-z][a-z-]*", " ".join(match.groups(default="")))) if match else ()
        )
        assert dependencies == JOB_PREREQUISITES[job]
        if "python-validation" in dependencies and job != "ci-gate":
            assert "needs.python-validation.result == 'success'" in header
        blocks = jobs[job].split("      - name: ")[1:]
        names = [block.splitlines()[0] for block in blocks]
        assert names == [check.name for check in checks] + [SUMMARY_NAME]
        assert len(names) == len(set(names))
        for check, block in zip(checks, blocks, strict=False):
            assert re.findall(r"^        id: (\w+)$", block, re.MULTILINE) == [check.step_id]
            explicit_dependencies = tuple(re.findall(r"steps\.(\w+)\.outcome == 'success'", block))
            if check.step_id in {"checkout", "setup", "classify", "filesystem", "evaluate"}:
                # These first checks use Actions' implicit success() after checkout.
                assert "        if:" not in block
                assert check.prerequisites == (() if check.step_id == "checkout" else ("checkout",))
            else:
                assert explicit_dependencies == check.prerequisites
                assert "!cancelled()" in block
            if check.command:
                assert f"        run: {check.command}\n" in block
            if check.branches:
                expression = (
                    f"matrix.suite == '{check.branches[0]}'"
                    if job == "contracts"
                    else f"matrix.shard == {check.branches[0]}"
                )
                assert expression in block
            else:
                condition = [line for line in block.splitlines() if line.startswith("        if:")]
                assert all("matrix." not in line for line in condition)


def test_workflow_matches_shared_names_ids_commands_and_prerequisites():
    assert_workflow_contract(_workflow())
    assert len(PROFILES) == 9
    assert set(HEAVY_JOBS) == {
        "python-validation",
        "quality",
        "contracts",
        "integration",
        "image-security",
    }
    assert set(FULL_JOB_REQUIRED_STEPS) == set(PROFILES)
    for checks in CHECKS.values():
        seen = set()
        for check in checks:
            assert set(check.prerequisites) <= seen
            seen.add(check.step_id)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("needs: [preflight, python-validation]", "needs: preflight"),
        ("      - name: Format check\n", "      - name: Renamed format\n"),
        ("        id: format\n", "        id: rules\n"),
        ("        run: make test-unit\n", "        run: make test-collect\n"),
        ("matrix.suite == 'eval'", "matrix.suite == 'operational'"),
        ("steps.build.outcome == 'success'", "steps.checkout.outcome == 'success'"),
    ],
)
def test_contract_rejects_workflow_drift(old, new):
    workflow = _workflow()
    assert old in workflow
    with pytest.raises(AssertionError):
        assert_workflow_contract(workflow.replace(old, new, 1))


def test_contract_rejects_missing_or_duplicate_step():
    workflow = _workflow()
    start = workflow.index("      - name: Format check\n")
    end = workflow.index("      - name: Lint rules\n", start)
    block = workflow[start:end]
    for changed in (workflow[:start] + workflow[end:], workflow[:start] + block + workflow[start:]):
        with pytest.raises(AssertionError):
            assert_workflow_contract(changed)


def test_make_selectors_execute_exactly_the_declared_routes():
    makefile = (ROOT / "Makefile").read_text()
    commands = {check.command for checks in CHECKS.values() for check in checks if check.command}
    for route in TEST_ROUTES:
        assert f"make {route.target}" in commands
        lines = makefile.split(f"\n{route.target}:", 1)[1].splitlines()
        recipe = [lines[0]]
        for line in lines[1:]:
            if line and not line[0].isspace():
                break
            recipe.append(line)
        body = "\n".join(recipe)
        if route.prefix:
            assert f"pytest {route.prefix.rstrip('/')}" in body
        else:
            assert route.node in body
        assert tuple(re.findall(r"--deselect (\S+)", body)) == route.excluded
    collect = makefile.split("\ntest-collect:", 1)[1].split("\n\n", 1)[0]
    assert 'PYTHONPATH="$(CURDIR)"' in collect
    assert "pytest -p tests.ci_coverage tests --collect-only -q" in collect
    assert "ci_coverage" not in (ROOT / "pyproject.toml").read_text()


def test_collection_assigns_all_categories_and_special_nodes_once():
    nodes = tuple(
        [
            f"tests/{category}/test_x.py::test_x"
            for category in ("unit", "architecture", "operational", "evals", "integration")
        ]
        + [RETRIEVAL_NODE, DEMO_NODE + "[parameter]"]
    )
    counts = collection_ownership(nodes)
    assert counts == {route.target: 1 for route in TEST_ROUTES}


@pytest.mark.parametrize("nodes", [(), ("tests/new/test_x.py::test_x",), ("same", "same")])
def test_empty_duplicate_and_unowned_tests_fail_closed(nodes):
    with pytest.raises(ValueError):
        collection_ownership(nodes)


def test_overlapping_routes_are_rejected_even_when_they_name_the_same_target():
    node = "tests/unit/test_x.py::test_x[PRIVATE-CANARY]"
    for target in ("test-unit", "other"):
        routes = (*TEST_ROUTES, Route(target, prefix="tests/unit/"))
        with pytest.raises(ValueError, match="exactly one execution target") as error:
            collection_ownership((node,), routes)
        assert "CANARY" not in str(error.value)
