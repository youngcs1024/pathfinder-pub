"""Manual E4.7 pilot operations. Never starts application services or accepts baselines."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from pydantic import SecretStr

from app.config import Settings
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.session import create_database_engine, create_session_factory
from app.domain.runs import CURRENT_GRAPH_VERSION
from app.llm.factory import LLMFactory
from app.llm.ports import LOCKED_CHAT_MODEL
from app.llm.pricing import QWEN_BEIJING_PRICING_VERSION
from app.llm.qwen_adapters import create_qwen_adapters
from app.obs.logging import configure_logging
from tests.evals.harness import _StrictMemoryInvocationRecorder
from tests.evals.live_baseline import PROJECT_ROOT, probe_clean_git_head
from tests.evals.quality_contracts import (
    QualityGenerationPolicyV1,
    QualityMappingV1,
    QualityRetrievalPolicyV1,
    QualityRunManifestV1,
)
from tests.evals.quality_dataset import (
    load_quality_dataset,
    prepare_quality_mapping_sources,
    quality_digest,
    quality_identity_digest,
    validate_quality_mapping,
)
from tests.evals.quality_generation import (
    GENERATION_SUITE_VERSION,
    generation_configuration_digest,
    generation_prompt_digest,
)
from tests.evals.quality_generation_support import checked_directory
from tests.evals.quality_review import (
    ReviewError,
    encoded,
    export_reviews,
    freeze_review,
    import_calibration,
    read_private,
    write_new,
)
from tests.evals.quality_run import run_quality_generation

PILOT_CASES = (
    "mixed_alpha",
    "mixed_beta",
    "strength_gamma",
    "multi_alpha",
    "multi_beta",
    "gap_alpha",
    "gap_gamma",
    "answer_alpha",
    "answer_beta",
    "web_alpha",
    "web_beta",
    "web_gamma",
    "distractor_beta",
    "injection_gamma",
    "scope_missing",
)
IMAGE = "pgvector/pgvector:0.8.5-pg16"
DATASET = PROJECT_ROOT / "evals/datasets/quality_v1"
MAPPING = PROJECT_ROOT / "evals/datasets/quality_mappings/pilot_v1_heading_v1"
CANDIDATE = PROJECT_ROOT / "tests/fixtures/quality_reviews/e47-rubric-v1/rubric.json"
DEFAULT_ROOT = Path.home() / "pathfinder-private" / "quality-e47"


class PilotError(ValueError):
    pass


def credentials(path: Path):
    try:
        values = {}
        for line in read_private(path, 8192).decode().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if (
                not separator
                or key not in {"DASHSCOPE_API_KEY", "PF_QWEN_WORKSPACE_ID"}
                or key in values
            ):
                raise ValueError("keys")
            value = value.strip()
            if value[:1] in {"'", '"'}:
                if value[-1:] != value[:1]:
                    raise ValueError("quotes")
                value = value[1:-1]
            if not value or any(c.isspace() for c in value):
                raise ValueError("empty")
            values[key] = SecretStr(value)
        if set(values) != {"DASHSCOPE_API_KEY", "PF_QWEN_WORKSPACE_ID"}:
            raise ValueError("missing")
        Settings.validate_qwen_workspace_id(values["PF_QWEN_WORKSPACE_ID"])
        return values
    except Exception:
        raise PilotError("credentials_missing_or_invalid") from None


def docker(*arguments: str, timeout: int = 60, env=None) -> str:
    try:
        result = subprocess.run(
            ("docker", *arguments), capture_output=True, text=True, timeout=timeout, env=env
        )
        if result.returncode:
            raise ValueError("docker")
        return result.stdout.strip()
    except Exception:
        raise PilotError("docker_operation_failed") from None


def preflight(root: Path):
    try:
        root = checked_directory(root)
        if root.is_relative_to(PROJECT_ROOT.resolve()) or PROJECT_ROOT.resolve().is_relative_to(
            root
        ):
            raise PilotError("private_root_overlaps_repository")
        values = credentials(root / "credentials.env")
        head = probe_clean_git_head()
        dataset = load_quality_dataset(DATASET)
        mapping = QualityMappingV1.model_validate_json((MAPPING / "mapping.json").read_bytes())
        rules = quality_digest((MAPPING / "rules.md").read_bytes())
        result = validate_quality_mapping(
            dataset, mapping, prepare_quality_mapping_sources(DATASET), rules_digest=rules
        )
        if (
            not result.mapping_complete
            or result.pending_review_chunks
            or not set(PILOT_CASES) <= {c.case_id for c in dataset.cases}
            or dataset.manifest.license_category != "synthetic"
        ):
            raise PilotError("pilot_material_invalid")
        docker("image", "inspect", IMAGE, "--format", "{{.Id}}")
        if docker("info", "--format", "{{.OSType}}") != "linux":
            raise PilotError("linux_docker_required")
        return values, head, dataset, mapping, rules
    except PilotError:
        raise
    except Exception:
        raise PilotError("pilot_preflight_failed") from None


def build_manifest(dataset, head, factory, experiment):
    policy = QualityGenerationPolicyV1(
        retrieval=QualityRetrievalPolicyV1(unknown_attempt_reserve_cny=Decimal("0.1"))
    )
    manifest = QualityRunManifestV1(
        experiment_id=experiment,
        execution_source_sha=head,
        suite_version=GENERATION_SUITE_VERSION,
        dataset_version=dataset.manifest.dataset_version,
        dataset_digest=dataset.manifest_digest,
        case_set_digest=quality_identity_digest(list(PILOT_CASES)),
        split_digest=quality_identity_digest(
            [[f.family_id, f.split] for f in dataset.manifest.families]
        ),
        rubric_version=dataset.rubric.rubric_version,
        rubric_digest=next(f.digest for f in dataset.manifest.files if f.role == "rubric"),
        model=LOCKED_CHAT_MODEL,
        prompt_digest=generation_prompt_digest(),
        graph_version=CURRENT_GRAPH_VERSION,
        embedding_profile=policy.retrieval.embedding_profile,
        retrieval_policy_digest=quality_identity_digest(policy.retrieval.model_dump(mode="json")),
        configuration_digest=generation_configuration_digest(policy, factory),
        measurement_scope="generation",
        llm_mode="qwen",
        web_mode="frozen_fixture",
        document_mode="real_embedding_db",
        selected_case_ids=PILOT_CASES,
        repeat_count=1,
        execution_order=tuple({"case_id": c, "repeat_index": 0} for c in PILOT_CASES),
        cost_admission_budget_cny=Decimal("10"),
        provider_attempt_cap=300,
        input_token_cap=1_000_000,
        output_token_cap=100_000,
    )
    return manifest, policy


def provision_database(container, database, password):
    environment = dict(os.environ, POSTGRES_PASSWORD=password)
    docker(
        "run",
        "--detach",
        "--name",
        container,
        "--label",
        "pathfinder.e47=owned",
        "--cpus",
        "2",
        "--memory",
        "2g",
        "--memory-swap",
        "2g",
        "--pids-limit",
        "256",
        "--publish",
        "127.0.0.1::5432",
        "--env",
        "POSTGRES_USER=pathfinder_test",
        "--env",
        "POSTGRES_PASSWORD",
        "--env",
        f"POSTGRES_DB={database}",
        IMAGE,
        env=environment,
    )
    ready = False
    for _ in range(60):
        try:
            docker(
                "exec", container, "pg_isready", "-U", "pathfinder_test", "-d", database, timeout=5
            )
            ready = True
            break
        except PilotError:
            time.sleep(1)
    if not ready:
        raise PilotError("database_not_ready")
    address = docker("port", container, "5432/tcp")
    host, separator, port = address.partition(":")
    if host != "127.0.0.1" or not separator or not port.isdigit():
        raise PilotError("database_binding_invalid")
    url = f"postgresql+psycopg://pathfinder_test:{password}@127.0.0.1:{port}/{database}"
    environment = dict(
        os.environ,
        PF_DATABASE_URL=url,
        PF_LLM_MODE="fake",
        PF_SEARCH_MODE="fake",
        PF_AUTH_MODE="fake",
        PF_TRACE_MODE="off",
    )
    result = subprocess.run(
        (sys.executable, "-m", "alembic", "upgrade", "head"),
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        timeout=120,
    )
    if result.returncode:
        raise PilotError("pilot_migration_failed")
    return url


def operation(path, **fields):
    write_new(path, (json.dumps(fields, sort_keys=True) + "\n").encode())


async def run_pilot(root: Path, *, confirm_live: bool):
    if not confirm_live:
        raise PilotError("live_confirmation_required")
    values, head, dataset, mapping, rules = preflight(root)
    for name in ("operations", "runs", "public", "reviews", "calibrations", "freezes"):
        directory = root / name
        directory.mkdir(mode=0o700, exist_ok=True)
        checked_directory(directory)
    suffix = uuid4().hex
    experiment, container, database = (
        "e47_" + suffix,
        "pf-e47-" + suffix,
        "pathfinder_test_" + suffix,
    )
    directory = root / "operations" / experiment
    directory.mkdir(mode=0o700)
    operation(
        directory / "start.json",
        experiment_id=experiment,
        source_sha=head,
        container=container,
        pricing_version=QWEN_BEIJING_PRICING_VERSION,
        live="NOT_RUN",
    )
    bundle = engine = None
    outcome = "failed"
    attempted_container = False
    try:
        bundle = create_qwen_adapters(
            api_key=values["DASHSCOPE_API_KEY"], workspace_id=values["PF_QWEN_WORKSPACE_ID"]
        )
        factory = LLMFactory(
            _StrictMemoryInvocationRecorder(), bundle.chat, bundle.embedding, provider="qwen"
        )
        manifest, policy = build_manifest(dataset, head, factory, experiment)
        write_new(
            directory / "manifest.json",
            encoded(manifest),
            markers=tuple(v.get_secret_value() for v in values.values()),
        )
        attempted_container = True
        url = provision_database(container, database, uuid4().hex)
        engine = create_database_engine(SecretStr(url))
        sessions = create_session_factory(engine)
        factory = LLMFactory(
            SqlAlchemyInvocationRecorder(sessions), bundle.chat, bundle.embedding, provider="qwen"
        )
        print(json.dumps({"stage": "generation", "experiment_id": experiment}), flush=True)
        report = await run_quality_generation(
            sessions,
            dataset_root=DATASET,
            dataset=dataset,
            mapping=mapping,
            mapping_rules_digest=rules,
            manifest=manifest,
            policy=policy,
            output_dir=root / "public" / experiment,
            private_root=root / "runs",
            provider_mode="qwen",
            confirm_disposable_database=True,
            confirm_live=True,
            factory=factory,
            sensitive_markers=tuple(v.get_secret_value() for v in values.values()),
        )
        operation(
            directory / "result.json",
            experiment_id=experiment,
            executed=report.executed,
            failed=report.failed,
            not_run=report.not_run,
            evidence_valid=report.evidence_valid,
            output_count=sum(c.observation.output_digest is not None for c in report.cases),
        )
        if report.evidence_valid:
            export_reviews(
                root / "runs" / experiment, DATASET, CANDIDATE, root / "reviews" / experiment
            )
        outcome = "finished"
        return experiment
    finally:
        cleanup_failed = False
        if engine is not None:
            try:
                await engine.dispose()
            except Exception:
                cleanup_failed = True
        if bundle is not None:
            try:
                await bundle.aclose()
            except Exception:
                cleanup_failed = True
        if attempted_container:
            try:
                # Only the unguessable name reserved by this invocation; no deletion or --rm.
                docker("stop", "--time", "10", container, timeout=30)
            except PilotError:
                cleanup_failed = True
        operation(
            directory / "stop.json",
            experiment_id=experiment,
            outcome=outcome,
            cleanup_failed=cleanup_failed,
            evidence_retained=True,
        )
        if cleanup_failed:
            print('{"stage":"cleanup","status":"failed"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Manual E4.7 calibration tools; no default live run"
    )
    parser.add_argument("command", choices=("preflight", "run", "export", "import", "freeze"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--private-run", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--confirm-human-stable", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(log_level="WARNING")
    try:
        if args.command == "preflight":
            preflight(args.root)
        elif args.command == "run":
            experiment = asyncio.run(run_pilot(args.root, confirm_live=args.confirm_live))
            print(json.dumps({"experiment_id": experiment}))
        elif args.command == "export":
            export_reviews(args.private_run, DATASET, args.candidate, args.output)
        elif args.command == "import":
            result = import_calibration(args.review_dir, DATASET, args.reviewer)
            write_new(args.output, encoded(result))
            print(
                json.dumps(
                    {
                        "complete": result.complete,
                        "reviewed_outputs": sum(
                            c.annotation.assessment_complete for c in result.cases
                        ),
                    }
                )
            )
        else:
            freeze_review(
                args.review_dir,
                DATASET,
                args.calibration,
                args.output,
                confirm_human_stable=args.confirm_human_stable,
            )
        return 0
    except (PilotError, ReviewError) as error:
        print(json.dumps({"status": "stopped", "category": str(error)}))
        return 2
    except (KeyboardInterrupt, asyncio.CancelledError):
        print('{"status":"stopped","category":"cancelled"}')
        return 130
    except Exception:
        print('{"status":"stopped","category":"pilot_operation_failed"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
