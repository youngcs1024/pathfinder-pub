"""E5.8 fixed scale selection and strictly body-free measurement evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from tests.performance.capacity_contracts import Health, Stop
from tests.performance.contracts import Digest, digest
from tests.performance.metrics import Distribution
from tests.performance.workload import Contract

AUTHORIZATION = "e58_local_retrieval_user_approved_v1"
type Kind = Literal["baseline", "workspaces", "candidates", "history", "global", "instant"]
type Phase = Literal["first_after_load", "warmup", "measurement"]
type Query = Literal[0, 1, 2, 3, 4]
type Table = Literal[
    "documents",
    "document_chunks",
    "workspaces",
    "workspace_memberships",
    "users",
    "conversations",
    "messages",
    "runs",
    "run_jobs",
    "run_events",
    "llm_invocations",
]


class Profile(Contract):
    schema_version: Literal[1] = 1
    name: Literal["retrieval-e58-v1", "retrieval-instant-ci-v1"]
    seed: Literal[58] = 58
    query_count: int
    warmup: int
    repeats: int
    max_seconds: Literal[240] = 240
    cleanup_seconds: Literal[70] = 70
    suite_seconds: Literal[2700] = 2700
    memory_limit: Literal[939524096] = 896 * 1024**2
    tmpfs_limit: Literal[234881024] = 224 * 1024**2
    rss_limit: Literal[2147483648] = 2 * 1024**3
    minimum_free: Literal[2147483648] = 2 * 1024**3
    connection_limit: Literal[32] = 32
    max_queue: Literal[1] = 1

    @model_validator(mode="after")
    def fixed(self):
        expected = (5, 2, 10) if self.name == "retrieval-e58-v1" else (2, 0, 2)
        if (self.query_count, self.warmup, self.repeats) != expected:
            raise ValueError("invalid_profile")
        return self

    @property
    def max_attempts(self):
        return self.query_count * (1 + self.warmup + self.repeats)


def load_profile(name):
    if name not in {"retrieval-e58-v1", "retrieval-instant-ci-v1"}:
        raise ValueError("invalid_profile")
    return Profile.model_validate_json(
        (Path(__file__).parent / "profiles" / f"{name}.json").read_bytes()
    )


class Point(Contract):
    kind: Kind
    documents: int
    chunks: int
    workspaces: int
    candidates: int
    events: int
    history_runs: int

    @model_validator(mode="after")
    def fixed(self):
        expected = {
            "baseline": (100, 1000, 10, 10, 1000, 100),
            "workspaces": (100, 1000, 50, 10, 1000, 100),
            "candidates": (100, 1000, 10, 100, 1000, 100),
            "history": (100, 1000, 10, 10, 10000, 100),
            "global": (1000, 10000, 10, 10, 1000, 100),
            "instant": (6, 30, 2, 5, 4, 2),
        }[self.kind]
        if (
            tuple(
                getattr(self, key)
                for key in (
                    "documents",
                    "chunks",
                    "workspaces",
                    "candidates",
                    "events",
                    "history_runs",
                )
            )
            != expected
        ):
            raise ValueError("invalid_point")
        return self


def points(instant=False):
    values = (
        (("instant", 6, 30, 2, 5, 4, 2),)
        if instant
        else (
            ("baseline", 100, 1000, 10, 10, 1000, 100),
            ("workspaces", 100, 1000, 50, 10, 1000, 100),
            ("candidates", 100, 1000, 10, 100, 1000, 100),
            ("history", 100, 1000, 10, 10, 10000, 100),
            ("global", 1000, 10000, 10, 10, 1000, 100),
        )
    )
    return tuple(
        Point(
            **dict(
                zip(
                    (
                        "kind",
                        "documents",
                        "chunks",
                        "workspaces",
                        "candidates",
                        "events",
                        "history_runs",
                    ),
                    row,
                    strict=True,
                )
            )
        )
        for row in values
    )


class Manifest(Contract):
    schema_version: Literal[1] = 1
    experiment_id: UUID
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_digest: Digest
    profile: Profile
    profile_digest: Digest
    point: Point
    authorization: Literal["e58_local_retrieval_user_approved_v1", "ci_instant"]
    database_image: Literal["pgvector/pgvector:0.8.5-pg16"]
    database_version: str = Field(pattern=r"^16\.[0-9]{1,3}$")
    vector_extension_version: str = Field(pattern=r"^[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}$")
    generator_digest: Digest
    query_set_digest: Digest
    generator_version: Literal["retrieval-scale-seed58-v1"] = "retrieval-scale-seed58-v1"
    distribution: Literal["dense_iid_uniform_minus1_plus1_float32"] = (
        "dense_iid_uniform_minus1_plus1_float32"
    )
    dimension: Literal[1536] = 1536
    embedding_profile: Literal["qwen-beijing-text-embedding-v4-1536-v1"] = (
        "qwen-beijing-text-embedding-v4-1536-v1"
    )
    top_k: Literal[5] = 5
    allowed_document_count: Literal[1] = 1
    worker_count: Literal[0] = 0
    database_cpu: Literal[1] = 1
    database_memory: Literal[1073741824] = 1024**3
    database_tmpfs: Literal[268435456] = 256 * 1024**2
    modes: tuple[Literal["fake", "off"], ...] = ("fake", "fake", "fake", "off")
    cache_condition: Literal["loaded_analyzed_os_cache_uncontrolled"] = (
        "loaded_analyzed_os_cache_uncontrolled"
    )
    history_basis: Literal["synthetic_storage_fixture_not_worker_execution"] = (
        "synthetic_storage_fixture_not_worker_execution"
    )
    provider_max_attempts: Literal[1] = 1

    @model_validator(mode="after")
    def identity(self):
        instant = self.authorization == "ci_instant"
        if (
            self.experiment_id.version != 4
            or digest(self.profile) != self.profile_digest
            or self.modes != ("fake", "fake", "fake", "off")
            or (self.point.kind == "instant") != instant
            or (self.profile.name == "retrieval-instant-ci-v1") != instant
        ):
            raise ValueError("identity_mismatch")
        return self


class Sample(Contract):
    query: Query
    phase: Phase
    repeat: int = Field(ge=0, lt=10)
    outcome: Literal["not_run", "succeeded", "failed", "timeout", "cancelled"] = "not_run"
    total_seconds: float | None = Field(default=None, ge=0)
    repository_seconds: float | None = Field(default=None, ge=0)
    sql_seconds: float | None = Field(default=None, ge=0)
    returned: int | None = Field(default=None, ge=0, le=5)
    sql_count: int = Field(default=0, ge=0, le=1)
    result_digest: Digest | None = None


class Relation(Contract):
    name: Table
    rows: int = Field(ge=0)
    table_bytes: int = Field(ge=0)
    index_bytes: int = Field(ge=0)
    total_bytes: int = Field(ge=0)


class Layout(Contract):
    candidates: int = Field(ge=0, le=100)
    target_workspace_chunks: int = Field(ge=0, le=120)
    workspace_chunk_counts: tuple[int, ...] = Field(max_length=50)
    document_chunk_counts: tuple[int, ...] = Field(max_length=1000)
    event_counts: tuple[int, ...] = Field(max_length=100)
    relations: tuple[Relation, ...] = Field(max_length=11)
    dataset_digest: Digest


type NodeType = Literal[
    "Limit",
    "Sort",
    "Incremental Sort",
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Heap Scan",
    "Bitmap Index Scan",
    "BitmapAnd",
    "BitmapOr",
    "Nested Loop",
    "Hash Join",
    "Merge Join",
    "Hash",
    "Materialize",
    "Memoize",
    "Gather",
    "Gather Merge",
]


class PlanNode(Contract):
    node: NodeType
    relation: Literal["documents", "document_chunks"] | None = None
    index: str | None = Field(
        default=None, pattern=r"^(?:pk|ix|uq|documents|document_chunks)_[a-z0-9_]{1,120}$"
    )
    values: dict[
        Literal[
            "Startup Cost",
            "Total Cost",
            "Plan Rows",
            "Plan Width",
            "Actual Startup Time",
            "Actual Total Time",
            "Actual Rows",
            "Actual Loops",
            "Rows Removed by Filter",
            "Rows Removed by Join Filter",
            "Shared Hit Blocks",
            "Shared Read Blocks",
            "Shared Dirtied Blocks",
            "Shared Written Blocks",
            "Temp Read Blocks",
            "Temp Written Blocks",
            "Sort Space Used",
            "Peak Memory Usage",
            "Heap Fetches",
        ],
        float,
    ]
    sort_method: (
        Literal["top-N heapsort", "quicksort", "external merge", "external sort"] | None
    ) = None
    sort_space: Literal["Memory", "Disk"] | None = None
    children: tuple[PlanNode, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def nonnegative(self):
        from tests.performance.retrieval_metrics import allowed_indexes

        if any(value < 0 for value in self.values.values()):
            raise ValueError("invalid_plan")
        if self.index is not None and self.index not in allowed_indexes():
            raise ValueError("invalid_plan_index")
        return self


class Plan(Contract):
    query: Query
    sql_digest: Digest
    planning_ms: float = Field(ge=0)
    execution_ms: float = Field(ge=0)
    root: PlanNode


class Summary(Contract):
    query: Query
    phase: Phase
    metric: Literal["total_seconds", "repository_seconds", "sql_seconds"]
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    not_run: int = Field(ge=0)
    latency: Distribution


class Result(Contract):
    schema_version: Literal[1] = 1
    point: Point
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    manifest_digest: Digest | None = None
    status: Literal["PASS", "IN_PROGRESS", "NOT_RUN"] = "IN_PROGRESS"
    stop: Stop
    stage: Literal[
        "environment",
        "generation",
        "analyze",
        "layout",
        "measurement",
        "plans",
        "reconcile",
        "finalizing",
    ] = "environment"
    diagnostics: tuple[Stop, ...] = ()
    samples: tuple[Sample, ...] = Field(default=(), max_length=65)
    summaries: tuple[Summary, ...] = Field(default=(), max_length=45)
    plans: tuple[Plan, ...] = Field(default=(), max_length=5)
    layout: Layout | None = None
    final_layout: Layout | None = None
    health: tuple[Health, ...] = Field(default=(), max_length=240)
    attempts: int | None = Field(default=None, ge=0, le=65)
    adapter_calls: int = Field(default=0, ge=0, le=65)
    observer_seconds: float = Field(default=0.0, ge=0)
    pool_remaining: int | None = Field(default=None, ge=0)
    resources_released: bool = False
    correctness: bool = False
    completeness: bool = False
    performance_target_met: None = None
    semantic_quality_claim: Literal[False] = False
    sql_latency_basis: Literal["driver_cursor_execute_excludes_fetch_and_explain"] = (
        "driver_cursor_execute_excludes_fetch_and_explain"
    )
    repository_latency_basis: Literal["membership_pool_transaction_sql_fetch_and_mapping"] = (
        "membership_pool_transaction_sql_fetch_and_mapping"
    )
    total_latency_basis: Literal["fake_embedding_factory_accounting_and_retrieval_service"] = (
        "fake_embedding_factory_accounting_and_retrieval_service"
    )

    @model_validator(mode="after")
    def honest(self):
        if self.status == "PASS" and not (
            self.stop == "window_complete"
            and self.completeness
            and self.correctness
            and self.resources_released
            and self.pool_remaining == 0
            and not self.diagnostics
            and self.layout is not None
            and self.final_layout is not None
            and self.samples
            and all(s.outcome == "succeeded" for s in self.samples)
            and self.attempts == self.adapter_calls == len(self.samples)
            and self.manifest_digest is not None
        ):
            raise ValueError("invalid_pass")
        if self.status == "PASS":
            from tests.performance.retrieval_metrics import check_layout, sample_schedule, summaries

            profile = load_profile(
                "retrieval-instant-ci-v1" if self.point.kind == "instant" else "retrieval-e58-v1"
            )
            expected = sample_schedule(profile)
            if (
                [(s.query, s.phase, s.repeat) for s in self.samples]
                != [(s.query, s.phase, s.repeat) for s in expected]
                or [p.query for p in self.plans] != list(range(profile.query_count))
                or self.summaries != summaries(self.samples)
                or any(
                    s.sql_count != 1
                    or s.returned != 5
                    or s.result_digest is None
                    or s.total_seconds is None
                    or s.repository_seconds is None
                    or s.sql_seconds is None
                    for s in self.samples
                )
            ):
                raise ValueError("incomplete_measurement")
            check_layout(self.point, self.layout)
            check_layout(self.point, self.final_layout)
        return self


class Suite(Contract):
    schema_version: Literal[1] = 1
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    authorization: Literal["e58_local_retrieval_user_approved_v1"]
    profile: Profile
    results: tuple[Result, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def identity(self):
        if (
            tuple(r.point for r in self.results) != points()
            or any(r.source_sha != self.source_sha for r in self.results)
            or self.profile.name != "retrieval-e58-v1"
        ):
            raise ValueError("identity_mismatch")
        return self


def validate_publication(name, record):
    if {
        "retrieval-manifest.json": Manifest,
        "retrieval-result.json": Result,
        "retrieval-suite.json": Suite,
    }.get(name) is not type(record):
        raise ValueError("invalid_retrieval_artifact")
