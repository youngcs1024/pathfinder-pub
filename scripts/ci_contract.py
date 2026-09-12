"""Standard-library-only CI contract; never imported by production application code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    step_id: str
    name: str
    prerequisites: tuple[str, ...] = ()
    branches: tuple[str, ...] = ()
    command: str | None = None

    def applies(self, branch: str) -> bool:
        return not self.branches or branch in self.branches


CHECKOUT = Check("checkout", "Check out repository")
SETUP = Check("setup", "Install locked Python environment", ("checkout",))
SUMMARY_NAME = "Summarize check outcomes"
CHECKS = {
    "preflight": (
        Check("checkout", "Check out repository history"),
        Check("classify", "Classify change set conservatively", ("checkout",)),
        Check("links", "Check public repository boundary", ("checkout",)),
        Check("secret_scan", "Scan repository history for secrets", ("checkout",)),
    ),
    "python-validation": (
        CHECKOUT,
        SETUP,
        Check("format", "Format check", ("setup",), command="make lint-format"),
        Check("rules", "Lint rules", ("setup",), command="make lint-rules"),
        Check("collect", "Collect all tests", ("setup",), command="make test-collect"),
        Check(
            "routing",
            "Validate real CI routing",
            ("setup", "collect"),
            command="make test-ci-routing",
        ),
    ),
    "quality": (
        CHECKOUT,
        SETUP,
        Check("setup_node", "Install locked JavaScript test runtime", ("checkout",)),
        Check("ui", "UI asynchronous behavior", ("setup_node",), command="make test-ui"),
        Check("unit", "Unit tests", ("setup",), command="make test-unit"),
        Check(
            "architecture", "Architecture constraints", ("setup",), command="make test-architecture"
        ),
        Check("audit", "Audit locked dependencies", ("setup",), command="make audit-deps"),
    ),
    "contracts": (
        CHECKOUT,
        SETUP,
        Check(
            "operational",
            "Operational contracts",
            ("setup",),
            ("operational",),
            "make test-operational-contracts",
        ),
        Check("eval", "Eval harness contracts", ("setup",), ("eval",), "make test-eval-contracts"),
        Check(
            "regression",
            "Accepted offline research regression",
            ("setup",),
            ("eval",),
            "make evals-regression",
        ),
    ),
    "integration": (
        CHECKOUT,
        SETUP,
        Check(
            "core", "PostgreSQL integration core", ("setup",), command="make test-integration-core"
        ),
        Check(
            "retrieval",
            "Deterministic PostgreSQL retrieval regression",
            ("setup",),
            ("0",),
            "make evals-retrieval",
        ),
        Check("demo", "Canonical offline product demo", ("setup",), ("0",), "make demo"),
    ),
    "image-security": (
        CHECKOUT,
        Check("filesystem", "Scan repository filesystem vulnerabilities", ("checkout",)),
        Check("build", "Build application image", ("checkout",), command="make image-build"),
        Check("smoke", "Smoke-test application image", ("build",), command="make image-smoke"),
        Check("library", "Scan application image library vulnerabilities", ("build",)),
        Check("os", "Scan application image OS vulnerabilities", ("build",)),
        Check("secret_scan", "Scan application image secrets", ("build",)),
        Check("image", "Record CI local image content ID", ("build",)),
    ),
    "ci-gate": (
        CHECKOUT,
        Check("evaluate", "Evaluate required CI results", ("checkout",)),
    ),
}
BRANCHES = {"contracts": ("operational", "eval"), "integration": ("0", "1")}
HEAVY_JOBS = tuple(job for job in CHECKS if job not in {"preflight", "ci-gate"})
JOB_PREREQUISITES = {
    "preflight": (),
    "python-validation": ("preflight",),
    "quality": ("preflight", "python-validation"),
    "contracts": ("preflight", "python-validation"),
    "integration": ("preflight", "python-validation"),
    "image-security": ("preflight",),
    "ci-gate": ("preflight", *HEAVY_JOBS),
}
INTEGRATION_SHARD_COUNT = 2
STEP_IDS = {job: tuple(check.step_id for check in checks) for job, checks in CHECKS.items()}
PROFILES = {
    job + (f" ({branch})" if branch else ""): (job, branch)
    for job in CHECKS
    for branch in BRANCHES.get(job, ("",))
}
FULL_JOB_REQUIRED_STEPS = {
    profile: (*(check.name for check in CHECKS[job] if check.applies(branch)), SUMMARY_NAME)
    for profile, (job, branch) in PROFILES.items()
}
FULL_JOB_SKIPPED_STEPS = {
    profile: tuple(check.name for check in CHECKS[job] if not check.applies(branch))
    for profile, (job, branch) in PROFILES.items()
    if any(not check.applies(branch) for check in CHECKS[job])
}

RETRIEVAL_NODE = (
    "tests/integration/db/test_retrieval_benchmark.py::"
    "test_real_db_benchmark_pipeline_filters_accounting_and_determinism"
)
DEMO_NODE = "tests/integration/db/test_gate8_demo.py::test_gate8_demo_complete_application_flow"


@dataclass(frozen=True)
class TestRoute:
    target: str
    prefix: str | None = None
    node: str | None = None
    excluded: tuple[str, ...] = ()

    def matches(self, node_id: str) -> bool:
        identity = node_id.split("[", 1)[0]
        return identity not in self.excluded and (
            (self.node is not None and identity == self.node)
            or (self.prefix is not None and identity.startswith(self.prefix))
        )


TEST_ROUTES = (
    TestRoute("test-unit", prefix="tests/unit/"),
    TestRoute("test-architecture", prefix="tests/architecture/"),
    TestRoute("test-operational-contracts", prefix="tests/operational/"),
    TestRoute("test-eval-contracts", prefix="tests/evals/"),
    TestRoute(
        "test-integration-core", prefix="tests/integration/", excluded=(RETRIEVAL_NODE, DEMO_NODE)
    ),
    TestRoute("evals-retrieval", node=RETRIEVAL_NODE),
    TestRoute("demo", node=DEMO_NODE),
)


def collection_ownership(
    node_ids: tuple[str, ...], routes: tuple[TestRoute, ...] = TEST_ROUTES
) -> dict[str, int]:
    """No parameter values or file contents may enter error messages or summaries."""
    if not node_ids or len(node_ids) != len(set(node_ids)):
        raise ValueError("empty or duplicate CI collection")
    counts = dict.fromkeys((route.target for route in routes), 0)
    for node_id in node_ids:
        matches = [route for route in routes if route.matches(node_id)]
        if len(matches) != 1:
            raise ValueError("CI test must have exactly one execution target")
        counts[matches[0].target] += 1
    return counts
