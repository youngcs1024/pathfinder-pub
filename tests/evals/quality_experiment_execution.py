"""Manual E7-A.7 paired generation, using the existing Factory/Registry/PG harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import Field
from sqlalchemy import select

from app.db.models import LLMInvocation, User
from app.llm.factory import LLMFactory
from app.llm.ports import LOCKED_CHAT_MODEL, LOCKED_EMBEDDING_MODEL
from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier
from tests.evals.live_suite import can_admit_attempt
from tests.evals.quality_contracts import (
    QualityGenerationPolicyV1,
    QualityMappingV1,
    QualityRetrievalAdmissionV1,
    QualityRetrievalPolicyV1,
    QualityRunManifestV1,
)
from tests.evals.quality_dataset import (
    load_quality_dataset,
    quality_digest,
    quality_identity_digest,
)
from tests.evals.quality_experiment import DATASET, load_experiment_plan
from tests.evals.quality_experiment_binding import (
    PLAN,
    ROOT,
    BindingV1,
    ExperimentError,
    build_binding,
    ci_proof,
    command,
    encoded,
    fail,
    no_links,
    probe_environment,
    read_json,
    source_identity,
    verify_binding,
    verify_imports,
    write_new,
)
from tests.evals.quality_experiment_database import OwnedExperimentDatabase, require_owned_database
from tests.evals.quality_experiment_resources import ResourceObserver
from tests.evals.quality_generation_support import checked_directory

ARMS = ("baseline", "candidate")


class _IdentityAdapter:
    """No HTTP or fabricated results; used solely to hash the existing Qwen profile."""

    provider = "qwen"

    def __init__(self, model):
        self.model = model

    async def invoke(self, *args, **kwargs):
        fail("identity_adapter_not_executable")

    async def embed(self, *args, **kwargs):
        fail("identity_adapter_not_executable")


def live_identity_factory(recorder):
    return LLMFactory(
        recorder,
        _IdentityAdapter(LOCKED_CHAT_MODEL),
        _IdentityAdapter(LOCKED_EMBEDDING_MODEL),
        provider="qwen",
    )


def configured_policy(root=ROOT):
    policy = QualityGenerationPolicyV1(
        retrieval=QualityRetrievalPolicyV1(unknown_attempt_reserve_cny=Decimal("0.1"))
    )
    plan = load_experiment_plan(root / PLAN, root=root)
    if (
        quality_identity_digest(policy.retrieval.model_dump(mode="json"))
        != plan.identity.retrieval_policy_digest
    ):
        fail("retrieval_policy_drift")
    return policy


def arm_manifest(binding, arm, factory, *, root=ROOT):
    if arm not in ARMS:
        fail("invalid_arm")
    from tests.evals.quality_generation import (
        generation_configuration_digest,
        generation_prompt_digest,
    )

    plan = load_experiment_plan(root / PLAN, root=root)
    identity = plan.identity
    policy = configured_policy(root)
    value = dict(
        experiment_id=f"{binding.experiment_id}_{arm}",
        execution_source_sha=getattr(binding, f"{arm}_source_sha"),
        suite_version="quality-generation-v1",
        dataset_version=identity.dataset_version,
        dataset_digest=identity.dataset_digest,
        split_digest=identity.split_digest,
        case_set_digest=identity.case_set_digest,
        rubric_version=identity.rubric_version,
        rubric_digest=identity.rubric_digest,
        model=identity.model,
        prompt_digest=generation_prompt_digest(arm=arm),
        graph_version=identity.baseline_graph_version
        if arm == "baseline"
        else binding.candidate_graph_version,
        embedding_profile=identity.embedding_profile,
        retrieval_policy_digest=identity.retrieval_policy_digest,
        configuration_digest=generation_configuration_digest(policy, factory, arm=arm),
        measurement_scope="generation",
        llm_mode="qwen",
        web_mode="frozen_fixture",
        document_mode="real_embedding_db",
        selected_case_ids=list(plan.samples.selected_case_ids),
        repeat_count=3,
        execution_order=[
            {"case_id": s.case_id, "repeat_index": s.repeat_index}
            for s in plan.samples.execution_order
            if s.arm == arm
        ],
        cost_admission_budget_cny=plan.budget.per_arm_cny,
        provider_attempt_cap=plan.budget.per_arm_provider_attempts,
        input_token_cap=plan.budget.per_arm_input_tokens,
        output_token_cap=plan.budget.per_arm_output_tokens,
    )
    manifest = QualityRunManifestV1.model_validate_json(encoded(value))
    expected = (
        identity.baseline_configuration_digest
        if arm == "baseline"
        else binding.candidate_configuration_digest
    )
    if manifest.configuration_digest != expected:
        fail("configuration_drift")
    return manifest, policy


class UsageV1(EvalContractModel):
    attempts: int = Field(ge=0)
    started: int = Field(ge=0)
    known_cost_cny: Decimal = Field(ge=0)
    unknown_cost_attempts: int = Field(ge=0)
    unknown_usage_attempts: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    invocation_digest: EvalDigest


def summarize_rows(rows):
    known = Decimal(0)
    unknown = unknown_usage = inputs = outputs = started = 0
    identities = []
    for row in rows:
        identities.append(str(row.id))
        started += row.status == "started"
        if row.estimated_cost is None:
            unknown += 1
        else:
            known += row.estimated_cost
        if row.token_usage is None:
            unknown_usage += 1
        else:
            inputs += row.token_usage["input_tokens"]
            outputs += row.token_usage["output_tokens"]
    return UsageV1(
        attempts=len(rows),
        started=started,
        known_cost_cny=known,
        unknown_cost_attempts=unknown,
        unknown_usage_attempts=unknown_usage,
        input_tokens=inputs,
        output_tokens=outputs,
        invocation_digest=quality_identity_digest(sorted(identities)),
    )


class ExperimentHooks:
    """Narrow trusted test context. PG remains the only invocation/cost ledger."""

    def __init__(self, directory, *, observer=None, source_check=None, deadline=None):
        self.directory, self.observer, self.source_check = directory, observer, source_check
        self.deadline = deadline
        self.owner = self.manifest = self.admission = self.tenant = None
        self.ordinal = 0
        self.failure = None
        self.last_usage = None

    def bind(self, handle, manifest, policy):
        if self.owner is not None:
            fail("budget_context_reused")
        self.owner = require_owned_database(handle)
        self.manifest = manifest
        self.admission = QualityRetrievalAdmissionV1.model_validate_json(
            encoded(
                {
                    **manifest.model_dump(mode="json"),
                    "unknown_attempt_reserve_cny": str(
                        policy.retrieval.unknown_attempt_reserve_cny
                    ),
                }
            )
        )

    async def usage(self, attempt=None):
        if self.owner is None or self.owner._closed:
            fail("invalid_budget_owner")
        async with self.owner._sessions() as session:
            if self.tenant is None:
                if attempt is None:
                    return summarize_rows([])
                subject = await session.scalar(
                    select(User.auth_subject).where(User.id == attempt.actor_user_id)
                )
                if subject != f"quality-{self.manifest.experiment_id}":
                    fail("budget_tenant_mismatch")
                self.tenant = (attempt.workspace_id, attempt.actor_user_id)
            if attempt is not None and self.tenant != (attempt.workspace_id, attempt.actor_user_id):
                fail("budget_tenant_mismatch")
            rows = list(
                await session.scalars(
                    select(LLMInvocation).where(
                        LLMInvocation.workspace_id == self.tenant[0],
                        LLMInvocation.actor_user_id == self.tenant[1],
                    )
                )
            )
        usage = summarize_rows(rows)
        self.last_usage = usage
        return usage

    async def before_attempt(self, attempt):
        try:
            if self.observer:
                self.observer.check()
            if self.deadline is not None and monotonic() >= self.deadline:
                fail("execution_deadline")
            if self.source_check:
                self.source_check()
            usage = await self.usage(attempt)
            if usage.started:
                fail("unfinished_provider_attempt")
            if not can_admit_attempt(
                self.admission,
                known_cost_cny=usage.known_cost_cny,
                unknown_cost_attempt_count=usage.unknown_cost_attempts,
                provider_attempts=usage.attempts,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ):
                fail("budget_exhausted")
            write_new(self.directory / f"admission-{self.ordinal:05d}.json", usage)
            self.ordinal += 1
        except Exception as error:
            self.failure = (
                error.category if isinstance(error, ExperimentError) else "accounting_failed"
            )
            raise

    async def after_attempt(self, attempt, outcome):
        try:
            usage = await self.usage(attempt)
            write_new(self.directory / f"accounted-{self.ordinal:05d}.json", usage)
            if (
                usage.started
                or usage.known_cost_cny
                + usage.unknown_cost_attempts * self.admission.unknown_attempt_reserve_cny
                > self.admission.cost_admission_budget_cny
                or usage.input_tokens > self.admission.input_token_cap
                or usage.output_tokens > self.admission.output_token_cap
                or usage.attempts > self.admission.provider_attempt_cap
            ):
                fail("budget_exhausted")
        except Exception as error:
            self.failure = (
                error.category if isinstance(error, ExperimentError) else "accounting_failed"
            )
            raise

    def record_slot(self, index, result, statistics):
        write_new(
            self.directory / f"diagnostic-{index:04d}.json",
            {
                "slot_index": index,
                "output_digest": result.observation.output_digest,
                "status": result.observation.status,
                "statistics": statistics,
            },
        )


class AuthorizationV1(EvalContractModel):
    artifact_kind: Literal["e7a7_run_authorization_v1"] = "e7a7_run_authorization_v1"
    binding_digest: EvalDigest
    source: Literal["user_explicit_e7a7_live_and_agent_review"]
    synthetic_materials: Literal[True]
    frozen_144_slots: Literal[True]
    per_arm_cny: Literal[30]
    total_cny: Literal[60]
    own_wsl_resources: Literal[True]
    agent_initial_and_recheck: Literal[True]
    production_adoption: Literal[False] = False
    deployment: Literal[False] = False


class SlotV1(EvalContractModel):
    ordinal: int = Field(ge=0, lt=144)
    arm: Literal["baseline", "candidate"]
    case_id: EvalIdentifier
    repeat_index: int = Field(ge=0, le=2)
    status: Literal["planned", "succeeded", "failed", "not_run", "missing"]
    case_digest: EvalDigest | None = None


class ExecutionV1(EvalContractModel):
    artifact_kind: Literal["e7a7_execution_v1"] = "e7a7_execution_v1"
    binding_digest: EvalDigest
    slots: tuple[SlotV1, ...] = Field(min_length=144, max_length=144)
    stop_category: str | None
    cleanup_complete: bool
    arm_report_digests: dict[Literal["baseline", "candidate"], EvalDigest]
    resource_report_digests: dict[Literal["baseline", "candidate"], EvalDigest]
    elapsed_seconds: float = Field(ge=0)
    execution_complete: bool
    measurement_scope: Literal["generation"] = "generation"


def planned_slots(plan):
    return tuple(
        SlotV1(
            ordinal=i, arm=s.arm, case_id=s.case_id, repeat_index=s.repeat_index, status="planned"
        )
        for i, s in enumerate(plan.samples.execution_order)
    )


def check_slot(expected, actual):
    if (
        actual.ordinal != expected.ordinal
        or actual.arm != expected.arm
        or (actual.case_id, actual.repeat_index) != (expected.case_id, expected.repeat_index)
    ):
        fail("slot_order_mismatch")


# The bootstrap executes before any app/tests import in a fresh -I interpreter.
BOOTSTRAP = """import sys, asyncio
sys.dont_write_bytecode = True
sys.pycache_prefix = sys.argv[5] + "/unwritten-bytecode"
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[2]]
from tests.evals.quality_experiment_execution import child_main
raise SystemExit(asyncio.run(child_main(*sys.argv[3:])))
"""


@dataclass
class Child:
    arm: str
    process: object

    async def send(self, value):
        self.process.stdin.write(encoded(value))
        await self.process.stdin.drain()

    async def receive(self, deadline, peers=()):
        pending = asyncio.create_task(self.process.stdout.readline())
        try:
            while not pending.done():
                if monotonic() >= deadline:
                    fail("execution_deadline")
                if any(p.process.returncode is not None for p in peers if p is not self):
                    fail("peer_process_stopped")
                await asyncio.wait({pending}, timeout=min(0.5, max(0, deadline - monotonic())))
            line = pending.result()
            if not line or len(line) > 65536:
                fail("child_protocol_failed")
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("category") not in {
                "ready",
                "prepared",
                "slot",
                "finished",
                "failed",
            }:
                fail("child_protocol_failed")
            if value["category"] == "failed":
                fail("child_execution_failed")
            return value
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def launch_child(binding_path, binding, arm, run_root, credentials_path):
    source = binding.baseline_root if arm == "baseline" else binding.candidate_root
    environment = {
        k: v
        for k, v in os.environ.items()
        if k in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    }
    environment.update(
        PF_AUTH_MODE="fake",
        PF_SEARCH_MODE="fake",
        PF_TRACE_MODE="off",
        TESTCONTAINERS_RYUK_DISABLED="true",
        PYTHONDONTWRITEBYTECODE="1",
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        BOOTSTRAP,
        source,
        binding.harness_root,
        str(binding_path),
        arm,
        str(run_root),
        str(credentials_path),
        str(os.getpid()),
        cwd=source,
        env=environment,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        limit=65536,
        start_new_session=True,
    )
    return Child(arm, process)


async def _input(observer=None):
    # Linux readiness avoids a blocked reader thread surviving cancellation/shutdown.
    loop = asyncio.get_running_loop()
    work = loop.create_future()
    buffer = bytearray()

    def readable():
        try:
            data = os.read(sys.stdin.fileno(), 65537)
            if not data:
                fail("child_protocol_failed")
            buffer.extend(data)
            if len(buffer) > 65536:
                fail("child_protocol_failed")
            if b"\n" in buffer:
                if buffer.count(b"\n") != 1 or not buffer.endswith(b"\n"):
                    fail("child_protocol_failed")
                loop.remove_reader(sys.stdin.fileno())
                work.set_result(bytes(buffer))
        except Exception as error:
            loop.remove_reader(sys.stdin.fileno())
            if not work.done():
                work.set_exception(error)

    loop.add_reader(sys.stdin.fileno(), readable)
    try:
        raw = await observer.protect(work) if observer else await work
        message = json.loads(raw)
        if not isinstance(message, dict):
            fail("child_protocol_failed")
        return message
    finally:
        loop.remove_reader(sys.stdin.fileno())
        if not work.done():
            work.cancel()


def _reply(category, **fields):
    sys.stdout.buffer.write(encoded({"category": category, **fields}))
    sys.stdout.buffer.flush()


async def child_main(binding_path, arm, run_root, credentials_path, parent_pid):
    manager = observer = state = bundle = hooks = None
    exit_code, stop = 0, None
    root = Path(run_root)
    me = asyncio.current_task()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, me.cancel)
    try:
        binding = read_json(Path(binding_path), BindingV1)
        if arm not in ARMS:
            fail("invalid_arm")
        source = Path(binding.baseline_root if arm == "baseline" else binding.candidate_root)
        harness = Path(binding.harness_root)

        def source_check():
            source_identity(source, getattr(binding, f"{arm}_source_sha"))
            source_identity(harness, binding.candidate_source_sha)
            verify_imports(source, harness)

        source_check()
        verify_binding(binding, environment=probe_environment())
        from app.llm.qwen_adapters import create_qwen_adapters
        from app.obs.logging import configure_logging
        from tests.evals.harness import _StrictMemoryInvocationRecorder
        from tests.evals.quality_pilot import credentials
        from tests.evals.quality_run import (
            finish_quality_generation,
            prepare_quality_generation,
            run_quality_generation_slot,
        )

        configure_logging(log_level="INFO", stream=sys.stderr)
        values = credentials(Path(credentials_path))
        markers = tuple(v.get_secret_value() for v in values.values())
        bundle = create_qwen_adapters(
            api_key=values["DASHSCOPE_API_KEY"], workspace_id=values["PF_QWEN_WORKSPACE_ID"]
        )
        factory = LLMFactory(
            _StrictMemoryInvocationRecorder(), bundle.chat, bundle.embedding, provider="qwen"
        )
        manifest, policy = arm_manifest(binding, arm, factory, root=harness)
        manager = OwnedExperimentDatabase()
        handle = await manager.__aenter__()
        owner = require_owned_database(handle)
        if owner._container.get_wrapped_container().image.id != binding.environment.image_id:
            fail("image_drift")
        _reply("ready")
        message = await _input()
        if message.get("command") != "prepare" or type(message.get("deadline")) is not float:
            fail("child_protocol_failed")
        observer = ResourceObserver(
            owner, root / arm / "resources", arm, parent_pid=int(parent_pid)
        )
        await observer.start()
        hooks = ExperimentHooks(
            root / arm / "accounting",
            observer=observer,
            source_check=source_check,
            deadline=message["deadline"],
        )
        dataset_root = harness / DATASET
        state = await observer.protect(
            prepare_quality_generation(
                owned_database=handle,
                arm=arm,
                dataset_root=dataset_root,
                dataset=load_quality_dataset(dataset_root),
                mapping=read_json(dataset_root / "mapping.json", QualityMappingV1),
                mapping_rules_digest=quality_digest(
                    (dataset_root / "mapping-rules.md").read_bytes()
                ),
                manifest=manifest,
                policy=policy,
                output_dir=root / "outputs" / arm,
                private_root=root / "private",
                provider_mode="qwen",
                confirm_live=True,
                factory=factory,
                git_probe=lambda: source_check() or getattr(binding, f"{arm}_source_sha"),
                sensitive_markers=markers,
                experiment_hooks=hooks,
            )
        )
        if state.stop:
            fail("generation_preparation_failed")
        _reply("prepared")
        while True:
            message = await _input(observer)
            command_ = message.get("command")
            if command_ in {"finish", "abort"}:
                if command_ == "abort":
                    state.stop = state.stop or "cancelled"
                break
            if command_ != "slot" or type(message.get("index")) is not int:
                fail("child_protocol_failed")
            index = message["index"]
            source_check()
            result = await observer.protect(run_quality_generation_slot(state, index))
            _reply(
                "slot",
                index=index,
                status=result.observation.status,
                case_digest=quality_identity_digest(result.model_dump(mode="json")),
                stop=hooks.failure or state.stop,
            )
        report = await finish_quality_generation(state)
        write_new(root / arm / "usage.json", await hooks.usage())
        _reply("finished", report_digest=quality_identity_digest(report.model_dump(mode="json")))
    except BaseException as error:
        stop = (
            "cancelled"
            if isinstance(error, asyncio.CancelledError)
            else error.category
            if isinstance(error, ExperimentError)
            else "execution_failed"
        )
        exit_code = 1
        if state is not None and not state.closed and not state.active:
            try:
                from tests.evals.quality_run import finish_quality_generation

                state.stop = state.stop or "integrity"
                state.cancellation = None
                await finish_quality_generation(state)
            except Exception:
                pass  # Original failure and missing report remain explicit below.
        if hooks is not None:
            try:
                write_new(root / arm / "failure-usage.json", await hooks.usage())
            except Exception:
                pass
        write_new(
            root / arm / "failure.json",
            {"category": stop, "budget_category": hooks.failure if hooks else None},
        )
        _reply("failed")
    finally:
        resource_ok = False
        if observer is not None and not observer.closed:
            try:
                resource_ok = (await observer.finish()).complete
            except Exception:
                resource_ok = False
        close_failed = False
        if bundle is not None:
            try:
                await bundle.aclose()
            except Exception:
                close_failed = True
        if manager is not None:
            await manager.aclose()
            close_failed |= manager.cleanup_failed
        write_new(
            root / arm / "cleanup.json",
            {
                "cleanup_complete": not close_failed,
                "resource_complete": resource_ok,
                "stop_category": stop,
            },
        )
    return exit_code


async def execute(binding_path, run_root, credentials_path, authorization_path):
    binding = read_json(binding_path, BindingV1)
    verify_binding(binding, environment=probe_environment())
    authorization = read_json(authorization_path, AuthorizationV1)
    binding_digest = quality_identity_digest(binding.model_dump(mode="json"))
    if authorization.binding_digest != binding_digest:
        fail("run_not_authorized")
    root = no_links(run_root)
    checked_directory(root.parent)
    if any(root.is_relative_to(Path(p)) for p in (binding.baseline_root, binding.candidate_root)):
        fail("private_root_required")
    root.mkdir(mode=0o700, exist_ok=False)
    for name in ("outputs", "private", "slots", *ARMS):
        (root / name).mkdir(mode=0o700)
    for arm in ARMS:
        for name in ("resources", "accounting"):
            (root / arm / name).mkdir(mode=0o700)
    plan = load_experiment_plan(Path(binding.harness_root) / PLAN, root=Path(binding.harness_root))
    slots = list(planned_slots(plan))
    write_new(
        root / "plan.json",
        {
            "binding_digest": binding_digest,
            "authorization_digest": quality_identity_digest(authorization.model_dump(mode="json")),
            "slots": [s.model_dump(mode="json") for s in slots],
        },
    )
    children, stop, start = [], None, monotonic()
    deadline = start + plan.resources.execution_window_seconds
    counts = dict.fromkeys(ARMS, 0)
    try:
        for arm in ARMS:
            child = await launch_child(binding_path, binding, arm, root, credentials_path)
            children.append(child)
            if (await child.receive(min(deadline, monotonic() + 180), children))[
                "category"
            ] != "ready":
                fail("child_protocol_failed")
        for child in children:
            await child.send({"command": "prepare", "deadline": deadline})
            if (await child.receive(deadline, children))["category"] != "prepared":
                fail("child_protocol_failed")
        for slot in tuple(slots):
            verify_binding(binding)
            child = next(c for c in children if c.arm == slot.arm)
            write_new(root / "slots" / f"start-{slot.ordinal:04d}.json", slot)
            slots[slot.ordinal] = slot.model_copy(update={"status": "missing"})
            index = counts[slot.arm]
            await child.send({"command": "slot", "index": index})
            response = await child.receive(min(deadline, monotonic() + 620), children)
            if response.get("category") != "slot" or response.get("index") != index:
                fail("slot_order_mismatch")
            result = SlotV1.model_validate_json(
                encoded(
                    {
                        **slot.model_dump(mode="json"),
                        "status": response["status"],
                        "case_digest": response["case_digest"],
                    }
                )
            )
            check_slot(slot, result)
            slots[slot.ordinal] = result
            write_new(root / "slots" / f"result-{slot.ordinal:04d}.json", result)
            counts[slot.arm] += 1
            if response.get("stop"):
                fail("generation_stopped")
        for child in children:
            await child.send({"command": "finish"})
            if (await child.receive(min(deadline, monotonic() + 60)))["category"] != "finished":
                fail("child_protocol_failed")
    except BaseException as error:
        stop = (
            "cancelled"
            if isinstance(error, asyncio.CancelledError)
            else error.category
            if isinstance(error, ExperimentError)
            else "execution_failed"
        )
    finally:
        for child in children:
            if child.process.returncode is None:
                try:
                    if stop:
                        child.process.send_signal(signal.SIGTERM)
                    await asyncio.wait_for(child.process.wait(), 30)
                except (TimeoutError, ProcessLookupError):
                    if child.process.returncode is None:
                        child.process.kill()
                        await child.process.wait()
        cleanup, reports, resources = True, {}, {}
        for arm in ARMS:
            try:
                clean = read_json(root / arm / "cleanup.json")
                cleanup &= clean.get("cleanup_complete") is True
                if clean.get("stop_category"):
                    stop = stop or "child_execution_failed"
            except ExperimentError:
                cleanup = False
            for path, destination in (
                (root / "outputs" / arm / "report.json", reports),
                (root / arm / "resources" / "report.json", resources),
            ):
                if path.is_file():
                    destination[arm] = quality_identity_digest(read_json(path))
        slots = [
            s.model_copy(update={"status": "not_run"}) if s.status == "planned" else s
            for s in slots
        ]
        report = ExecutionV1(
            binding_digest=binding_digest,
            slots=tuple(slots),
            stop_category=stop,
            cleanup_complete=cleanup,
            arm_report_digests=reports,
            resource_report_digests=resources,
            elapsed_seconds=monotonic() - start,
            execution_complete=stop is None
            and cleanup
            and set(reports) == set(ARMS)
            and set(resources) == set(ARMS)
            and all(s.status in {"succeeded", "failed"} for s in slots),
        )
        write_new(root / "execution.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manual, frozen E7-A.7 experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    bind = commands.add_parser("experiment-bind")
    bind.add_argument("--baseline", type=Path, required=True)
    bind.add_argument("--candidate", type=Path, required=True)
    bind.add_argument("--experiment-id", required=True)
    bind.add_argument("--output", type=Path, required=True)
    bind.add_argument("--ci-evidence", type=Path, required=True)
    run = commands.add_parser("experiment-run")
    run.add_argument("--binding", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--credentials", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    for name in ("experiment-review-export", "experiment-review-import", "experiment-decide"):
        command_ = commands.add_parser(name)
        command_.add_argument("--binding", type=Path, required=True)
        command_.add_argument("--run-root", type=Path, required=True)
        command_.add_argument("--review-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "experiment-bind":
            summary = read_json(args.ci_evidence)
            supplied = ci_proof(summary)
            fresh = json.loads(
                command(
                    [
                        sys.executable,
                        str(args.candidate / "scripts/ci_inspect_run.py"),
                        "--repo",
                        "youngcs1024/pathfinder-pub",
                        "--run-id",
                        str(supplied.run_id),
                        "--expected-sha",
                        supplied.source_sha,
                        "--attempt",
                        str(supplied.attempt),
                    ]
                )
            )
            proof = ci_proof(fresh)
            if proof != supplied:
                fail("ci_evidence_drift")
            binding = build_binding(
                baseline_root=args.baseline,
                candidate_root=args.candidate,
                experiment_id=args.experiment_id,
                environment=probe_environment(),
                ci=proof,
            )
            write_new(args.output, binding)
            print(
                json.dumps(
                    {
                        "category": "binding_created",
                        "digest": quality_identity_digest(binding.model_dump(mode="json")),
                    }
                )
            )
        elif args.command == "experiment-run":
            report = asyncio.run(
                execute(args.binding, args.run_root, args.credentials, args.authorization)
            )
            print(
                json.dumps(
                    {
                        "category": "execution_complete"
                        if report.execution_complete
                        else "execution_incomplete"
                    }
                )
            )
            return 0 if report.execution_complete else 1
        else:
            from tests.evals.quality_experiment_decision import decision_command

            return decision_command(args)
        return 0
    except Exception as error:
        category = error.category if isinstance(error, ExperimentError) else "experiment_failed"
        print(json.dumps({"category": category}))
        return 1
