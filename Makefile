UV ?= uv
COMPOSE ?= docker compose
DEV_COMPOSE = PF_PUBLIC_DOMAIN=localhost $(COMPOSE) -f compose.yaml -f compose.dev.yaml
DOCKER ?= docker
IMAGE_TAG ?= pathfinder-ci:local
EXPECTED_UV_VERSION := 0.11.32
EXPECTED_PYTHON_VERSION := 3.12.13
PYTEST_TMPDIR ?= /tmp

.PHONY: verify-toolchain verify-toolchain-quiet lock-check sync lint test test-unit test-architecture test-operational-contracts test-eval-contracts test-integration-core test-integration audit-deps evals evals-regression evals-retrieval evals-baseline-refresh demo live-smoke live-evals-chat live-evals-chat-accepted live-evals-retrieval live-evals-retrieval-accepted live-evals-baseline-accept image-build image-smoke up down migrate run-api run-worker

lock-check:
	@command -v "$(UV)" >/dev/null 2>&1 || { \
		printf 'error: uv executable not found: %s\n' "$(UV)" >&2; \
		exit 1; \
	}
	@actual="$$( $(UV) --version 2>/dev/null | awk 'NR == 1 {print $$2; exit}' )"; \
	if [ "$$actual" != "$(EXPECTED_UV_VERSION)" ]; then \
		printf 'error: expected uv %s, got %s\n' "$(EXPECTED_UV_VERSION)" "$${actual:-unavailable}" >&2; \
		exit 1; \
	fi
	$(UV) lock --check

verify-toolchain: lock-check
	@actual="$$( $(UV) run --locked python -c 'import platform; print(platform.python_version())' )"; \
	if [ "$$actual" != "$(EXPECTED_PYTHON_VERSION)" ]; then \
		printf 'error: expected Python %s, got %s\n' "$(EXPECTED_PYTHON_VERSION)" "$${actual:-unavailable}" >&2; \
		exit 1; \
	fi
	@implementation="$$( $(UV) run --locked python -c 'import platform; print(platform.python_implementation())' )"; \
	if [ "$$implementation" != "CPython" ]; then \
		printf 'error: expected CPython, got %s\n' "$${implementation:-unavailable}" >&2; \
		exit 1; \
	fi

verify-toolchain-quiet:
	@command -v "$(UV)" >/dev/null 2>&1 || { \
		printf 'error: uv executable not found: %s\n' "$(UV)" >&2; \
		exit 1; \
	}
	@actual="$$( $(UV) --version 2>/dev/null | awk 'NR == 1 {print $$2; exit}' )"; \
	if [ "$$actual" != "$(EXPECTED_UV_VERSION)" ]; then \
		printf 'error: expected uv %s, got %s\n' "$(EXPECTED_UV_VERSION)" "$${actual:-unavailable}" >&2; \
		exit 1; \
	fi
	@$(UV) lock --check >/dev/null
	@actual="$$( $(UV) run --locked python -c 'import platform; print(platform.python_version())' )"; \
	if [ "$$actual" != "$(EXPECTED_PYTHON_VERSION)" ]; then \
		printf 'error: expected Python %s, got %s\n' "$(EXPECTED_PYTHON_VERSION)" "$${actual:-unavailable}" >&2; \
		exit 1; \
	fi
	@implementation="$$( $(UV) run --locked python -c 'import platform; print(platform.python_implementation())' )"; \
	if [ "$$implementation" != "CPython" ]; then \
		printf 'error: expected CPython, got %s\n' "$${implementation:-unavailable}" >&2; \
		exit 1; \
	fi

sync: lock-check
	$(UV) sync --locked
	@actual="$$( $(UV) run --locked python -c 'import platform; print(platform.python_version())' )"; \
	if [ "$$actual" != "$(EXPECTED_PYTHON_VERSION)" ]; then \
		printf 'error: expected Python %s, got %s\n' "$(EXPECTED_PYTHON_VERSION)" "$${actual:-unavailable}" >&2; \
		exit 1; \
	fi

lint: verify-toolchain
	@lint_status=0; \
	$(UV) run --locked ruff format --check . || lint_status=$$?; \
	$(UV) run --locked ruff check . || lint_status=$$?; \
	exit "$$lint_status"

.PHONY: lint-format lint-rules test-collect test-ci-routing
lint-format: verify-toolchain
	$(UV) run --locked ruff format --check .

lint-rules: verify-toolchain
	$(UV) run --locked ruff check .

test-collect: verify-toolchain
	PYTHONPATH="$(CURDIR)" TMPDIR="$(PYTEST_TMPDIR)" $(UV) run --locked pytest -p tests.ci_coverage tests --collect-only -q

test-ci-routing: verify-toolchain
	PF_CI_UV="$(UV)" $(UV) run --locked python scripts/ci_validate_routing.py

test: verify-toolchain
	@if [ ! -d "$(PYTEST_TMPDIR)" ] || [ ! -w "$(PYTEST_TMPDIR)" ]; then \
		printf 'error: pytest temporary directory is not writable: %s\n' \
			"$(PYTEST_TMPDIR)" >&2; \
		exit 1; \
	fi
	TMPDIR="$(PYTEST_TMPDIR)" $(UV) run --locked pytest

test-unit: verify-toolchain
	@if [ ! -d "$(PYTEST_TMPDIR)" ] || [ ! -w "$(PYTEST_TMPDIR)" ]; then \
		printf 'error: pytest temporary directory is not writable: %s\n' \
			"$(PYTEST_TMPDIR)" >&2; \
		exit 1; \
	fi
	TMPDIR="$(PYTEST_TMPDIR)" $(UV) run --locked pytest tests/unit

test-architecture: verify-toolchain
	@if [ ! -d "$(PYTEST_TMPDIR)" ] || [ ! -w "$(PYTEST_TMPDIR)" ]; then \
		printf 'error: pytest temporary directory is not writable: %s\n' \
			"$(PYTEST_TMPDIR)" >&2; \
		exit 1; \
	fi
	TMPDIR="$(PYTEST_TMPDIR)" $(UV) run --locked pytest tests/architecture

test-operational-contracts: verify-toolchain
	@if [ ! -d "$(PYTEST_TMPDIR)" ] || [ ! -w "$(PYTEST_TMPDIR)" ]; then \
		printf 'error: pytest temporary directory is not writable: %s\n' \
			"$(PYTEST_TMPDIR)" >&2; \
		exit 1; \
	fi
	TMPDIR="$(PYTEST_TMPDIR)" $(UV) run --locked pytest tests/operational

test-eval-contracts: verify-toolchain
	@if [ ! -d "$(PYTEST_TMPDIR)" ] || [ ! -w "$(PYTEST_TMPDIR)" ]; then \
		printf 'error: pytest temporary directory is not writable: %s\n' \
			"$(PYTEST_TMPDIR)" >&2; \
		exit 1; \
	fi
	TMPDIR="$(PYTEST_TMPDIR)" $(UV) run --locked pytest tests/evals

test-integration-core: verify-toolchain
	@if [ ! -d "$(PYTEST_TMPDIR)" ] || [ ! -w "$(PYTEST_TMPDIR)" ]; then \
		printf 'error: pytest temporary directory is not writable: %s\n' \
			"$(PYTEST_TMPDIR)" >&2; \
		exit 1; \
	fi
	TMPDIR="$(PYTEST_TMPDIR)" $(UV) run --locked pytest tests/integration \
		--deselect tests/integration/db/test_retrieval_benchmark.py::test_real_db_benchmark_pipeline_filters_accounting_and_determinism \
		--deselect tests/integration/db/test_gate8_demo.py::test_gate8_demo_complete_application_flow

test-integration: verify-toolchain
	@if [ ! -d "$(PYTEST_TMPDIR)" ] || [ ! -w "$(PYTEST_TMPDIR)" ]; then \
		printf 'error: pytest temporary directory is not writable: %s\n' \
			"$(PYTEST_TMPDIR)" >&2; \
		exit 1; \
	fi
	TMPDIR="$(PYTEST_TMPDIR)" $(UV) run --locked pytest tests/integration

audit-deps: lock-check
	$(UV) --preview-features audit-command audit --frozen

evals: export PF_EVAL_CASE := $(CASE)
evals:
	@command -v "$(UV)" >/dev/null 2>&1 || { \
		printf 'error: uv executable not found: %s\n' "$(UV)" >&2; \
		exit 1; \
	}
	@actual="$$( $(UV) --version 2>/dev/null | awk 'NR == 1 {print $$2; exit}' )"; \
	if [ "$$actual" != "$(EXPECTED_UV_VERSION)" ]; then \
		printf 'error: expected uv %s, got %s\n' "$(EXPECTED_UV_VERSION)" "$${actual:-unavailable}" >&2; \
		exit 1; \
	fi
	@$(UV) lock --check >/dev/null
	@actual="$$( $(UV) run --locked python -c 'import platform; print(platform.python_version())' )"; \
	if [ "$$actual" != "$(EXPECTED_PYTHON_VERSION)" ]; then \
		printf 'error: expected Python %s, got %s\n' "$(EXPECTED_PYTHON_VERSION)" "$${actual:-unavailable}" >&2; \
		exit 1; \
	fi
	@$(UV) run --locked python -m tests.evals

.PHONY: gate12-native-baseline
gate12-native-baseline: verify-toolchain-quiet
	@$(UV) run --locked python -m tests.evals.gate12_native

.PHONY: gate12-comparison
gate12-comparison: export UV_OFFLINE := true
gate12-comparison: verify-toolchain-quiet
	@$(UV) run --locked python -m tests.evals.gate12_compare

evals-regression: verify-toolchain-quiet
	@$(UV) run --locked python -m tests.evals.regression

evals-retrieval: verify-toolchain
	TMPDIR="$(PYTEST_TMPDIR)" $(UV) run --locked pytest -q -s \
		tests/integration/db/test_retrieval_benchmark.py::test_real_db_benchmark_pipeline_filters_accounting_and_determinism

evals-baseline-refresh: verify-toolchain-quiet
	@if [ -z "$(strip $(REASON))" ]; then \
		printf 'error: REASON is required for explicit baseline acceptance\n' >&2; \
		exit 2; \
	fi
	@$(UV) run --locked python -m tests.evals.baseline_refresh --reason "$(REASON)"

demo: verify-toolchain
	@if [ ! -d "$(PYTEST_TMPDIR)" ] || [ ! -w "$(PYTEST_TMPDIR)" ]; then \
		printf 'error: pytest temporary directory is not writable: %s\n' \
			"$(PYTEST_TMPDIR)" >&2; \
		exit 1; \
	fi
	PF_LLM_MODE=fake PF_SEARCH_MODE=fake PF_AUTH_MODE=fake PF_TRACE_MODE=off \
		PF_GATE87_DEMO_FAILURE="$(DEMO_FAILURE)" TMPDIR="$(PYTEST_TMPDIR)" \
		$(UV) run --locked pytest -q tests/integration/db/test_gate8_demo.py::test_gate8_demo_complete_application_flow

live-smoke: verify-toolchain
	@$(UV) run --locked python -m tests.evals --live-smoke

.PHONY: live-evals-chat
live-evals-chat: verify-toolchain-quiet
	@$(UV) run --locked python -m tests.evals --live-chat

live-evals-chat-accepted: verify-toolchain-quiet
	@$(UV) run --locked python -m tests.evals --live-chat-accepted

live-evals-retrieval: verify-toolchain-quiet
	PF_RUN_GATE11_LIVE_RETRIEVAL=1 TMPDIR="$(PYTEST_TMPDIR)" \
		$(UV) run --locked pytest -q -s \
		tests/integration/db/test_live_retrieval.py::test_real_qwen_embedding_retrieval_benchmark

live-evals-retrieval-accepted: verify-toolchain-quiet
	@if [ -z "$(strip $(REPORT))" ]; then \
		printf 'error: REPORT is required for create-only accepted retrieval output\n' >&2; \
		exit 2; \
	fi
	PF_RUN_GATE11_LIVE_RETRIEVAL_ACCEPTED=1 \
		PF_GATE11_ACCEPTED_RETRIEVAL_REPORT="$(REPORT)" TMPDIR="$(PYTEST_TMPDIR)" \
		$(UV) run --locked pytest -q -s \
		tests/integration/db/test_live_retrieval.py::test_accepted_qwen_embedding_retrieval_benchmark

live-evals-baseline-accept: verify-toolchain-quiet
	@if [ -z "$(strip $(CHAT_REPORT))" ] || [ -z "$(strip $(RETRIEVAL_REPORT))" ] || [ -z "$(strip $(REASON))" ]; then \
		printf 'error: CHAT_REPORT, RETRIEVAL_REPORT, and REASON are required\n' >&2; \
		exit 2; \
	fi
	@$(UV) run --locked python -m tests.evals.live_baseline \
		--chat-report "$(CHAT_REPORT)" --retrieval-report "$(RETRIEVAL_REPORT)" \
		--reason "$(REASON)"

image-build:
	$(DOCKER) build --tag "$(IMAGE_TAG)" .

image-smoke:
	PF_PUBLIC_DOMAIN=localhost PATHFINDER_IMAGE="$(IMAGE_TAG)" $(COMPOSE) run --rm --no-deps --entrypoint python api -c 'import importlib.util; import os; import platform; import sys; import tempfile; from pathlib import Path; import app; import app.api; import app.main; import app.worker.main; assert platform.python_version() == "$(EXPECTED_PYTHON_VERSION)"; assert Path(app.__file__).is_relative_to(Path(sys.prefix)); assert (Path(app.api.__file__).parent / "static" / "pathfinder.js").is_file(); assert all(importlib.util.find_spec(name) is None for name in ("pytest", "ruff", "testcontainers", "mcp")); assert os.geteuid() == 10001; assert os.getegid() == 10001; assert os.statvfs("/").f_flag & os.ST_RDONLY; temporary_file = tempfile.TemporaryFile(dir="/tmp"); temporary_file.write(b"pathfinder"); temporary_file.close(); mount_lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines(); tmp_mount = next(line for line in mount_lines if line.split()[4] == "/tmp"); assert tmp_mount.split(" - ", 1)[1].split()[0] == "tmpfs"; status = {key: value.strip() for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines() for key, _, value in (line.partition(":"),)}; assert int(status["CapEff"], 16) == 0; assert status["NoNewPrivs"] == "1"'

up:
	$(DEV_COMPOSE) up --detach --wait --wait-timeout 60 postgres

down:
	$(DEV_COMPOSE) down --timeout 10

migrate: verify-toolchain
	$(UV) run --locked alembic upgrade head

run-api: verify-toolchain
	$(UV) run --locked python -m app.main

run-worker: verify-toolchain
	$(UV) run --locked python -m app.worker.main

# Dev-only JavaScript behavior tests; no npm install or production build step.
.PHONY: test-ui
test-ui:
	@test "$$(node --version)" = "v$$(cat .node-version)" || { \
		printf 'error: Node version must match .node-version\n' >&2; exit 1; \
	}
	node --test tests/ui/*.test.cjs

# Workflow edits require this semantic check before committing; Python lint stays independent.
.PHONY: prepare-workflow-lint lint-workflow
prepare-workflow-lint:
	python3 scripts/workflow_lint.py prepare

lint-workflow:
	python3 scripts/workflow_lint.py check
