import re
from pathlib import Path

from scripts.ci_contract import CHECKS, HEAVY_JOBS, JOB_PREREQUISITES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
PYTHON_SETUP = PROJECT_ROOT / ".github" / "actions" / "setup-python" / "action.yml"
CHECKOUT_ACTION = "actions/checkout@9f698171ed81b15d1823a05fc7211befd50c8ae0"
SETUP_UV_ACTION = "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b"
GITLEAKS_ACTION = "gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e"
TRIVY_ACTION = "aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25"

REQUIRED_JOBS = tuple(CHECKS)


def _workflow() -> str:
    return CI_WORKFLOW.read_text(encoding="utf-8")


def _jobs(workflow: str) -> dict[str, str]:
    lines = workflow.splitlines(keepends=True)
    jobs: dict[str, list[str]] = {}
    current: str | None = None
    in_jobs = False
    for line in lines:
        if line == "jobs:\n":
            in_jobs = True
            continue
        match = re.fullmatch(r"  ([a-z][a-z0-9-]*):\n", line) if in_jobs else None
        if match:
            current = match.group(1)
            jobs[current] = [line]
        elif current is not None:
            jobs[current].append(line)
    return {name: "".join(body) for name, body in jobs.items()}


def test_ci_workflow_is_pinned_least_privilege_and_offline_by_default() -> None:
    workflow = _workflow()
    setup = PYTHON_SETUP.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert 'version: "0.11.32"' in setup
    assert 'python-version: "3.12.13"' in setup
    assert "uv lock --check" in setup
    assert "uv sync --locked" in setup
    assert "PF_LLM_MODE: fake" in workflow
    assert "PF_SEARCH_MODE: fake" in workflow
    assert "PF_AUTH_MODE: fake" in workflow
    assert "PF_TRACE_MODE: off" in workflow
    assert "pull_request_target" not in workflow
    assert "secrets." not in workflow
    assert "\n    services:" not in workflow

    for business_secret in (
        "DASHSCOPE_API_KEY",
        "TAVILY_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "PF_SUPABASE_PUBLISHABLE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    ):
        assert business_secret not in workflow

    external_uses = re.findall(r"uses: ([^\s]+@[^\s]+)", workflow + setup)
    assert external_uses
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", value) for value in external_uses)
    for action in (CHECKOUT_ACTION, SETUP_UV_ACTION, GITLEAKS_ACTION, TRIVY_ACTION):
        assert action in workflow + setup


def test_ci_workflow_has_parallel_full_ci_dag_and_docs_only_shortcut() -> None:
    workflow = _workflow()
    jobs = _jobs(workflow)

    assert set(jobs) == set(REQUIRED_JOBS)
    assert all("runs-on: ubuntu-24.04" in jobs[name] for name in REQUIRED_JOBS)
    assert "fetch-depth: 0" in jobs["preflight"]
    assert "scripts/ci_classify_changes.py" in jobs["preflight"]
    assert GITLEAKS_ACTION in jobs["preflight"]
    assert "outputs:\n      run_full:" in jobs["preflight"]

    for name in HEAVY_JOBS:
        dependencies = JOB_PREREQUISITES[name]
        expected = (
            dependencies[0] if len(dependencies) == 1 else "[" + ", ".join(dependencies) + "]"
        )
        assert f"needs: {expected}" in jobs[name]
        if "python-validation" in dependencies:
            assert "needs.python-validation.result == 'success'" in jobs[name]
        assert "if: needs.preflight.outputs.run_full == 'true'" in jobs[name]

    gate = jobs["ci-gate"]
    assert "if: always()" in gate
    assert "scripts/ci_gate.py" in gate
    for name in ("preflight", *HEAVY_JOBS):
        assert f"- {name}" in gate


def test_ci_workflow_runs_every_required_full_ci_command_in_its_semantic_job() -> None:
    jobs = _jobs(_workflow())

    expected = {
        "python-validation": (
            "make lint-format",
            "make lint-rules",
            "make test-collect",
            "make test-ci-routing",
        ),
        "quality": (
            "make audit-deps",
            "make test-unit",
            "make test-architecture",
            "make test-ui",
        ),
        "contracts": (
            "make test-operational-contracts",
            "make test-eval-contracts",
            "make evals-regression",
        ),
        "integration": (
            "make test-integration-core",
            "make evals-retrieval",
            "make demo",
        ),
        "image-security": ("make image-build", "make image-smoke"),
    }
    for job, commands in expected.items():
        for command in commands:
            assert command in jobs[job]

    workflow = _workflow()
    assert "run: make evals\n" not in workflow
    assert "evals-baseline-refresh" not in workflow
    assert "live-smoke" not in workflow
    assert "live-evals-chat" not in workflow


def test_ci_workflow_concurrency_preserves_manual_runs() -> None:
    workflow = _workflow()

    assert "concurrency:" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "format('manual-{0}', github.run_id)" in workflow
    assert "format('pr-{0}', github.event.pull_request.number)" in workflow
    assert "format('ref-{0}', github.ref)" in workflow
    assert "cancel-in-progress: ${{ github.event_name != 'workflow_dispatch' }}" in workflow


def _steps(job: str) -> dict[str, str]:
    return {block.splitlines()[0]: block for block in job.split("      - name: ")[1:]}


def test_public_boundary_runs_in_preflight_for_both_ci_paths() -> None:
    workflow = _workflow()
    preflight = _jobs(workflow)["preflight"]
    step = _steps(preflight)["Check public repository boundary"]
    assert "run: python scripts/check_public_repository.py\n" in step
    assert "if: ${{ !cancelled() && steps.checkout.outcome == 'success' }}" in step
    assert "continue-on-error" not in step
    assert workflow.count("run: python scripts/check_public_repository.py\n") == 1
    assert (
        "if: ${{ !cancelled() && steps.checkout.outcome == 'success' }}"
        in (_steps(preflight)["Scan repository history for secrets"])
    )


def test_contract_matrix_runs_each_required_suite_once_without_hiding_failure() -> None:
    job = _jobs(_workflow())["contracts"]
    assert "fail-fast: false" in job
    assert "max-parallel: 2" in job
    assert "suite: [operational, eval]" in job
    steps = _steps(job)
    for name, suite, target in (
        ("Operational contracts", "operational", "test-operational-contracts"),
        ("Eval harness contracts", "eval", "test-eval-contracts"),
        ("Accepted offline research regression", "eval", "evals-regression"),
    ):
        assert (
            "if: ${{ !cancelled() && steps.setup.outcome == 'success' && "
            f"matrix.suite == '{suite}' " + "}}"
        ) in steps[name]
        assert f"run: make {target}\n" in steps[name]
        assert job.count(f"run: make {target}\n") == 1


def test_integration_matrix_shards_only_core_and_runs_demo_and_retrieval_once() -> None:
    workflow = _workflow()
    job = _jobs(workflow)["integration"]
    assert "fail-fast: false" in job
    assert "max-parallel: 2" in job
    assert "shard: [0, 1]" in job
    assert "ci-shard" not in job.split("    steps:", 1)[0]
    steps = _steps(job)
    core = steps["PostgreSQL integration core"]
    assert "if: ${{ !cancelled() && steps.setup.outcome == 'success' }}" in core
    assert "          PYTEST_ADDOPTS: >-" in core
    assert "--durations=20 -p tests.ci_sharding" in core
    assert "--ci-shard-index=${{ matrix.shard }} --ci-shard-count=2" in core
    assert "          PYTHONPATH: ${{ github.workspace }}" in core
    assert workflow.count("PYTHONPATH:") == 2
    assert "run: make test-integration-core\n" in core
    assert workflow.count("-p tests.ci_sharding") == 1
    for name, target in (
        ("Deterministic PostgreSQL retrieval regression", "evals-retrieval"),
        ("Canonical offline product demo", "demo"),
    ):
        assert (
            "if: ${{ !cancelled() && steps.setup.outcome == 'success' && matrix.shard == 0 }}"
        ) in steps[name]
        assert "PYTEST_ADDOPTS" not in steps[name]
        assert "PYTHONPATH" not in steps[name]
        assert "ci-shard" not in steps[name]
        assert workflow.count(f"run: make {target}\n") == 1
    assert "PYTEST_ADDOPTS: --durations=20" in workflow.split("\njobs:", 1)[0]


def test_ci_workflow_security_scanners_are_pinned_and_fail_closed() -> None:
    workflow = _workflow()
    image_job = _jobs(workflow)["image-security"]
    preflight = _jobs(workflow)["preflight"]

    assert GITLEAKS_ACTION in preflight
    assert "GITHUB_TOKEN: ${{ github.token }}" in preflight
    assert "GITLEAKS_ENABLE_COMMENTS: false" in preflight
    assert "GITLEAKS_ENABLE_UPLOAD_ARTIFACT: false" in preflight

    trivy_steps = image_job.split(f"uses: {TRIVY_ACTION}")[1:]
    assert len(trivy_steps) == 4
    filesystem_scan, image_library_scan, image_os_scan, image_secret_scan = trivy_steps
    assert "scan-type: fs" in filesystem_scan
    assert "scan-ref: ." in filesystem_scan
    assert "scanners: vuln" in filesystem_scan
    assert "ignore-unfixed: false" in filesystem_scan

    for image_scan in (image_library_scan, image_os_scan, image_secret_scan):
        assert "scan-type: image" in image_scan
        assert "image-ref: ${{ env.IMAGE_TAG }}" in image_scan

    assert "vuln-type: library" in image_library_scan
    assert "ignore-unfixed: false" in image_library_scan
    assert "vuln-type: os" in image_os_scan
    assert "ignore-unfixed: true" in image_os_scan
    assert "scanners: secret" in image_secret_scan
    assert "ignore-unfixed: true" not in image_secret_scan
    assert image_job.count("ignore-unfixed: true") == 1

    for scan in trivy_steps:
        assert "severity: HIGH,CRITICAL" in scan
        assert "exit-code: 1" in scan
        assert "format: json" in scan
        assert "version: v0.70.0" in scan

    assert "exit-code: 0" not in workflow
    assert "continue-on-error: true" not in workflow
    assert "|| true" not in workflow


def test_independent_checks_depend_only_on_setup_not_previous_check_success():
    jobs = _jobs(_workflow())
    for job in ("python-validation", "quality", "contracts", "integration"):
        steps = _steps(jobs[job])
        assert "id: setup" in steps["Install locked Python environment"]
        assert "if:" not in steps["Install locked Python environment"]
        for name, step in steps.items():
            if name in {
                "Check out repository",
                "Install locked Python environment",
                "Install locked JavaScript test runtime",
                "UI asynchronous behavior",
                "Summarize check outcomes",
                "Validate safe diagnostics",
                "Upload safe diagnostics",
            }:
                continue
            condition = next(line.strip() for line in step.splitlines() if "if:" in line)
            assert "!cancelled()" in condition
            assert "steps.setup.outcome == 'success'" in condition
            assert "success()" not in condition
            assert condition.count("steps.") == (2 if name == "Validate real CI routing" else 1)
            assert "continue-on-error" not in step


def test_image_checks_require_build_but_not_previous_scan_or_smoke_success():
    steps = _steps(_jobs(_workflow())["image-security"])
    assert "id: checkout" in steps["Check out repository"]
    build = steps["Build application image"]
    assert "id: build" in build
    assert "if: ${{ !cancelled() && steps.checkout.outcome == 'success' }}" in build
    for name in (
        "Smoke-test application image",
        "Scan application image library vulnerabilities",
        "Scan application image OS vulnerabilities",
        "Scan application image secrets",
        "Record CI local image content ID",
    ):
        assert "if: ${{ !cancelled() && steps.build.outcome == 'success' }}" in steps[name]
        assert "continue-on-error" not in steps[name]


def test_every_job_reports_fixed_step_outcomes_after_failure_or_cancellation():
    from scripts.ci_step_summary import STEP_IDS

    for job, body in _jobs(_workflow()).items():
        steps = _steps(body)
        summary = steps["Summarize check outcomes"]
        assert list(steps)[-1] == "Summarize check outcomes"
        assert "if: ${{ always() && steps.checkout.outcome == 'success' }}" in summary
        assert f"PF_CI_JOB: {job}\n" in summary
        assert "PF_CI_STEPS: >-" in summary
        assert "toJSON(steps)" not in summary
        assert ".outputs" not in summary
        assert "run: python scripts/ci_step_summary.py\n" in summary
        condition = next(line for line in summary.splitlines() if line.strip().startswith("if:"))
        assert "steps.setup" not in condition
        assert "continue-on-error" not in summary
        ids = re.findall(r"^        id: (\w+)$", body, re.MULTILINE)
        assert len(ids) == len(set(ids))
        assert tuple(ids) == STEP_IDS[job]
        projected = re.findall(r"toJSON\(steps\.(\w+)\.(\w+)\)", summary)
        assert projected == [
            (step_id, field) for step_id in STEP_IDS[job] for field in ("outcome", "conclusion")
        ]
        expected_branch = {
            "contracts": "${{ matrix.suite }}",
            "integration": "${{ matrix.shard }}",
        }.get(job, "")
        assert f'PF_CI_BRANCH: "{expected_branch}"' in summary


def test_collection_and_lint_are_separate_independent_checks():
    steps = _steps(_jobs(_workflow())["python-validation"])
    for name, command in (
        ("Collect all tests", "make test-collect"),
        ("Format check", "make lint-format"),
        ("Lint rules", "make lint-rules"),
    ):
        assert f"run: {command}\n" in steps[name]
        assert "if: ${{ !cancelled() && steps.setup.outcome == 'success' }}" in steps[name]
    assert list(steps).index("Collect all tests") < list(steps).index("Validate real CI routing")


def test_static_feedback_precedes_runtime_tests_and_network_audit():
    steps = _steps(_jobs(_workflow())["quality"])
    assert list(steps) == [
        "Check out repository",
        "Install locked Python environment",
        "Install locked JavaScript test runtime",
        "UI asynchronous behavior",
        "Unit tests",
        "Architecture constraints",
        "Audit locked dependencies",
        "Validate safe diagnostics",
        "Upload safe diagnostics",
        "Summarize check outcomes",
    ]


def test_baseline_step_contract_matches_every_workflow_step_and_matrix_branch():
    from scripts.ci_classify_changes import FULL_JOB_REQUIRED_STEPS, FULL_JOB_SKIPPED_STEPS

    profiles = {
        "preflight": ("preflight", set()),
        "python-validation": ("python-validation", set()),
        "quality": ("quality", set()),
        "contracts (operational)": ("contracts", {"eval", "regression"}),
        "contracts (eval)": ("contracts", {"operational"}),
        "integration (0)": ("integration", set()),
        "integration (1)": ("integration", {"retrieval", "demo"}),
        "image-security": ("image-security", set()),
        "ci-gate": ("ci-gate", set()),
    }
    assert set(FULL_JOB_REQUIRED_STEPS) == set(profiles)
    assert set(FULL_JOB_SKIPPED_STEPS) == {key for key, (_, skip) in profiles.items() if skip}
    jobs = _jobs(_workflow())
    for profile, (job, inactive_ids) in profiles.items():
        required, skipped = [], []
        for name, body in _steps(jobs[job]).items():
            step_id = re.search(r"^        id: (\w+)$", body, re.MULTILINE)
            target = skipped if step_id and step_id[1] in inactive_ids else required
            target.append(name)
        assert tuple(required) == FULL_JOB_REQUIRED_STEPS[profile]
        assert tuple(skipped) == FULL_JOB_SKIPPED_STEPS.get(profile, ())


def test_summary_environment_is_valid_json_after_fixed_status_projection():
    import json

    from scripts.ci_step_summary import STEP_IDS, step_summary

    for job, body in _jobs(_workflow()).items():
        summary = _steps(body)["Summarize check outcomes"]
        projection = summary.split("PF_CI_STEPS: >-\n", 1)[1].split("        run:", 1)[0]
        branches = {"contracts": ("operational", "eval"), "integration": ("0", "1")}.get(job, ("",))
        for branch in branches:
            inactive = set()
            if job == "contracts":
                inactive = {"eval", "regression"} if branch == "operational" else {"operational"}
            elif job == "integration" and branch == "1":
                inactive = {"retrieval", "demo"}

            def replace_status(match, skipped=inactive):
                return json.dumps("skipped" if match.group(1) in skipped else "success")

            payload = re.sub(
                r"\$\{\{ toJSON\(steps\.(\w+)\.(outcome|conclusion)\) \}\}",
                replace_status,
                projection,
            )
            assert set(json.loads(payload)) == set(STEP_IDS[job])
            _, passed = step_summary(job, branch, payload)
            assert passed


def test_lint_annotations_are_scoped_to_ci_rule_check():
    workflow = _workflow()
    steps = _steps(_jobs(workflow)["python-validation"])
    assert "env:\n          RUFF_OUTPUT_FORMAT: github" in steps["Lint rules"]
    assert workflow.count("RUFF_OUTPUT_FORMAT") == 1
    assert "RUFF_OUTPUT_FORMAT" not in (PROJECT_ROOT / "Makefile").read_text()


def test_all_independent_checks_use_only_actual_prerequisites_and_preserve_failure():
    workflow = _workflow()
    expected = {
        "preflight": {
            "Check public repository boundary": "checkout",
            "Scan repository history for secrets": "checkout",
        },
        "python-validation": dict.fromkeys(
            ("Collect all tests", "Format check", "Lint rules"), "setup"
        ),
        "quality": dict.fromkeys(
            (
                "Audit locked dependencies",
                "Unit tests",
                "Architecture constraints",
            ),
            "setup",
        ),
        "contracts": dict.fromkeys(
            (
                "Operational contracts",
                "Eval harness contracts",
                "Accepted offline research regression",
            ),
            "setup",
        ),
        "integration": dict.fromkeys(
            (
                "PostgreSQL integration core",
                "Deterministic PostgreSQL retrieval regression",
                "Canonical offline product demo",
            ),
            "setup",
        ),
        "image-security": {
            "Build application image": "checkout",
            "Smoke-test application image": "build",
            "Scan application image library vulnerabilities": "build",
            "Scan application image OS vulnerabilities": "build",
            "Scan application image secrets": "build",
            "Record CI local image content ID": "build",
        },
    }
    jobs = _jobs(workflow)
    for job, checks in expected.items():
        steps = _steps(jobs[job])
        for name, prerequisite in checks.items():
            condition = next(line.strip() for line in steps[name].splitlines() if "if:" in line)
            assert "!cancelled()" in condition
            assert re.findall(r"steps\.(\w+)\.outcome == 'success'", condition) == [prerequisite]
            assert "success()" not in condition and "failure()" not in condition
            assert "||" not in condition
    assert "continue-on-error" not in workflow
    assert "|| true" not in workflow and "exit 0" not in workflow


def test_docs_shortcut_exposes_verified_evidence_and_scopes_actions_read_permission():
    workflow = _workflow()
    jobs = _jobs(workflow)
    assert workflow.count("actions: read") == 1
    assert "permissions:\n      contents: read\n      actions: read" in jobs["preflight"]
    assert "actions: write" not in workflow
    classify = _steps(jobs["preflight"])["Classify change set conservatively"]
    assert "GH_TOKEN: ${{ github.token }}" in classify
    assert workflow.count("GH_TOKEN:") == 1
    assert "--sha '${{ github.sha }}'" in classify
    for field in ("baseline_sha", "baseline_run_id", "baseline_attempt"):
        assert f"{field}: ${{{{ steps.classify.outputs.{field} }}}}" in jobs["preflight"]
        assert (
            f"--{field.replace('_', '-')} '${{{{ needs.preflight.outputs.{field} }}}}'"
            in jobs["ci-gate"]
        )


def test_ui_runtime_is_pinned_and_test_is_independent_of_python_checks():
    steps = _steps(_jobs(_workflow())["quality"])
    setup = steps["Install locked JavaScript test runtime"]
    ui = steps["UI asynchronous behavior"]
    assert "actions/setup-node@249970729cb0ef3589644e2896645e5dc5ba9c38" in setup
    assert "node-version-file: .node-version" in setup
    assert (PROJECT_ROOT / ".node-version").read_text().strip() == "24.20.0"
    assert "package-manager-cache: false" in setup
    assert "if: ${{ !cancelled() && steps.checkout.outcome == 'success' }}" in setup
    assert "if: ${{ !cancelled() && steps.setup_node.outcome == 'success' }}" in ui
    assert "run: make test-ui\n" in ui
    assert _workflow().count("run: make test-ui\n") == 1
    assert "continue-on-error" not in setup + ui


def test_safe_diagnostic_upload_is_bounded_and_preserves_original_failures():
    for job, body in _jobs(_workflow()).items():
        steps = _steps(body)
        validate = steps["Validate safe diagnostics"]
        upload = steps["Upload safe diagnostics"]
        for step in (validate, upload):
            assert "!cancelled() && steps.checkout.outcome == 'success'" in step
            assert "continue-on-error" not in step
        assert "python -m scripts.ci_reports" in validate
        assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in upload
        assert "path: ${{ runner.temp }}/pf-diagnostics/diagnostics.json" in upload
        assert "retention-days: 7" in upload and "if-no-files-found: error" in upload
        assert "${{ github.run_id }}-${{ github.run_attempt }}-" + job in upload
        assert "**" not in upload
    assert "-p tests.ci_reports" in _workflow().split("\njobs:")[0]
