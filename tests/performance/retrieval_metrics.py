"""Observe the production SELECT; parameters and raw plans never leave memory."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from time import monotonic
from typing import get_args

from sqlalchemy import event, func, select, text
from sqlalchemy.dialects import postgresql

from app.db.models import Document, DocumentChunk, Run, RunEvent
from app.retrieval.documents import EMBEDDING_PROFILE
from tests.performance.metrics import Value, distribution, missing
from tests.performance.retrieval_contracts import (
    Layout,
    Plan,
    PlanNode,
    Relation,
    Sample,
    Summary,
    Table,
)


def sample_schedule(profile):
    return [
        Sample(query=query, phase=phase, repeat=repeat)
        for query in range(profile.query_count)
        for phase, count in (
            ("first_after_load", 1),
            ("warmup", profile.warmup),
            ("measurement", profile.repeats),
        )
        for repeat in range(count)
    ]


def summaries(samples):
    result = []
    for query, phase in sorted({(s.query, s.phase) for s in samples}):
        group = [s for s in samples if (s.query, s.phase) == (query, phase)]
        counts = Counter(s.outcome for s in group)
        for metric in ("total_seconds", "repository_seconds", "sql_seconds"):
            values = [
                Value(value=getattr(s, metric))
                if getattr(s, metric) is not None
                else missing("not_run" if s.outcome == "not_run" else "unfinished")
                for s in group
            ]
            result.append(
                Summary(
                    query=query,
                    phase=phase,
                    metric=metric,
                    succeeded=counts["succeeded"],
                    not_run=counts["not_run"],
                    failed=len(group) - counts["succeeded"] - counts["not_run"],
                    latency=distribution(values),
                )
            )
    return tuple(result)


def allowed_indexes():
    preparer = postgresql.dialect().identifier_preparer
    return {
        preparer.format_constraint(c).strip('"')
        for model in (Document, DocumentChunk)
        for c in model.__table__.constraints
        if c.name is not None
    } | {
        preparer.format_index(i).strip('"')
        for model in (Document, DocumentChunk)
        for i in model.__table__.indexes
    }


def sanitize_plan(raw, *, query, sql_digest):
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise ValueError("invalid_plan")
    count = 0
    allowed = allowed_indexes()
    numeric = set(get_args(PlanNode.model_fields["values"].annotation)[0].__args__)

    def node(value, depth=0):
        nonlocal count
        count += 1
        if count > 64 or depth > 16 or not isinstance(value, dict):
            raise ValueError("invalid_plan")
        index = value.get("Index Name")
        if index is not None and index not in allowed:
            raise ValueError("invalid_plan_index")
        numbers = {key: value[key] for key in numeric if key in value}
        if any(type(v) not in {float, int} for v in numbers.values()):
            raise ValueError("invalid_plan_value")
        return PlanNode(
            node=value["Node Type"],
            relation=value.get("Relation Name"),
            index=index,
            values={key: float(v) for key, v in numbers.items()},
            sort_method=value.get("Sort Method"),
            sort_space=value.get("Sort Space Type"),
            children=tuple(node(child, depth + 1) for child in value.get("Plans", ())),
        )

    return Plan(
        query=query,
        sql_digest=sql_digest,
        planning_ms=raw[0]["Planning Time"],
        execution_ms=raw[0]["Execution Time"],
        root=node(raw[0]["Plan"]),
    )


class QueryObserver:
    """Engine-local listeners, scoped to one sequential repository invocation."""

    def __init__(self):
        self.active = False
        self.sql_seconds = None
        self.sql_count = 0
        self.repository_seconds = None
        self.captured = None
        self.started = None

    def before(self, connection, cursor, statement, parameters, context, many):
        if not self.active:
            return
        compiled = context.compiled
        selection = getattr(compiled, "statement", None)
        if not getattr(selection, "is_select", False):
            return
        if "cosine_distance" not in selection.selected_columns.keys():
            return
        joins = selection.get_final_froms()
        if (
            many
            or len(joins) != 1
            or getattr(getattr(joins[0], "left", None), "name", None) != "document_chunks"
            or getattr(getattr(joins[0], "right", None), "name", None) != "documents"
            or not statement.lstrip().startswith("SELECT ")
        ):
            raise ValueError("invalid_retrieval_select")
        self.sql_count += 1
        if self.sql_count > 1:
            raise ValueError("repeated_retrieval_select")
        self.captured = (statement, parameters)
        self.started = monotonic()

    def after(self, connection, cursor, statement, parameters, context, many):
        if self.active and self.started is not None:
            self.sql_seconds = monotonic() - self.started
            self.started = None

    @contextmanager
    def attach(self, engine):
        event.listen(engine.sync_engine, "before_cursor_execute", self.before)
        event.listen(engine.sync_engine, "after_cursor_execute", self.after)
        try:
            yield
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", self.before)
            event.remove(engine.sync_engine, "after_cursor_execute", self.after)
            self.captured = None

    async def search(self, repository, **kwargs):
        self.active = True
        self.sql_seconds, self.sql_count, self.started = None, 0, None
        self.captured = None
        started = monotonic()
        try:
            return await repository.search(**kwargs)
        finally:
            self.repository_seconds = monotonic() - started
            self.active = False


class ObservedRepository:
    def __init__(self, repository, observer):
        self.repository, self.observer = repository, observer

    async def search(self, **kwargs):
        return await self.observer.search(self.repository, **kwargs)


async def explain(engine, query, captured):
    statement, parameters = captured
    # Captured only from the actual production SELECT above. No SQL is read from a file
    # or accepted through the CLI. Parameters are passed through the driver's binding API.
    async with engine.connect() as connection, connection.begin():
        await connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        raw = (
            await connection.exec_driver_sql(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + statement, parameters
            )
        ).scalar_one()
    return sanitize_plan(
        raw, query=query, sql_digest=hashlib.sha256(statement.encode()).hexdigest()
    )


async def layout(sessions, dataset):
    tenant = dataset.tenants[0]
    async with sessions() as session, session.begin():
        await session.execute(text("SET TRANSACTION READ ONLY"))
        relations = []
        for name in get_args(Table.__value__):
            # Table names come solely from the closed type above, never from a profile/input.
            rows = await session.scalar(text(f'SELECT count(*) FROM "{name}"'))
            sizes = (
                await session.execute(
                    text(
                        "SELECT pg_table_size(CAST(:name AS regclass)), "
                        "pg_indexes_size(CAST(:name AS regclass)), "
                        "pg_total_relation_size(CAST(:name AS regclass))"
                    ),
                    {"name": name},
                )
            ).one()
            relations.append(
                Relation(
                    name=name,
                    rows=rows,
                    table_bytes=sizes[0],
                    index_bytes=sizes[1],
                    total_bytes=sizes[2],
                )
            )
        doc_counts = dict(
            (
                await session.execute(
                    select(DocumentChunk.document_id, func.count()).group_by(
                        DocumentChunk.document_id
                    )
                )
            ).all()
        )
        workspace_counts = dict(
            (
                await session.execute(
                    select(DocumentChunk.workspace_id, func.count()).group_by(
                        DocumentChunk.workspace_id
                    )
                )
            ).all()
        )
        event_rows = (
            await session.execute(
                select(
                    RunEvent.run_id,
                    func.count(),
                    func.min(RunEvent.seq),
                    func.max(RunEvent.seq),
                    Run.next_event_seq,
                )
                .join(
                    Run, (Run.id == RunEvent.run_id) & (Run.workspace_id == RunEvent.workspace_id)
                )
                .group_by(RunEvent.run_id, Run.next_event_seq)
            )
        ).all()
        if any(
            low != 1 or high != count or next_seq != high + 1
            for _, count, low, high, next_seq in event_rows
        ):
            raise ValueError("invalid_event_sequence")
        events = {row[0]: row[1] for row in event_rows}
        candidate_count = await session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .join(
                Document,
                (Document.workspace_id == DocumentChunk.workspace_id)
                & (Document.id == DocumentChunk.document_id)
                & (Document.embedding_model == DocumentChunk.embedding_model),
            )
            .where(
                Document.workspace_id == tenant.workspace_id,
                DocumentChunk.workspace_id == tenant.workspace_id,
                DocumentChunk.document_id == dataset.documents[0],
                Document.embedding_model == EMBEDDING_PROFILE,
                DocumentChunk.embedding_model == EMBEDDING_PROFILE,
            )
        )
    return Layout(
        candidates=candidate_count,
        target_workspace_chunks=workspace_counts.get(tenant.workspace_id, 0),
        workspace_chunk_counts=tuple(
            workspace_counts.get(t.workspace_id, 0) for t in dataset.tenants
        ),
        document_chunk_counts=tuple(doc_counts.get(d, 0) for d in dataset.documents),
        event_counts=tuple(events.get(r, 0) for r in dataset.history),
        relations=tuple(relations),
        dataset_digest=dataset.dataset_digest,
    )


def check_layout(point, current):
    from tests.performance.retrieval_data import allocations

    counts = {r.name: r.rows for r in current.relations}
    expected = {
        "documents": point.documents,
        "document_chunks": point.chunks,
        "workspaces": point.workspaces,
        "users": point.workspaces,
        "workspace_memberships": point.workspaces,
        "run_events": point.events,
        **{name: point.history_runs for name in ("conversations", "messages", "runs", "run_jobs")},
    }
    allocated = allocations(point)
    if (
        any(counts.get(name) != count for name, count in expected.items())
        or current.candidates != point.candidates
        or current.document_chunk_counts != tuple(count for _, count in allocated)
        or current.workspace_chunk_counts
        != tuple(
            sum(count for workspace, count in allocated if workspace == i)
            for i in range(point.workspaces)
        )
        or current.event_counts != (point.events // point.history_runs,) * point.history_runs
    ):
        raise ValueError("layout_mismatch")


def hit_digest(hits, dataset):
    if len(hits) != 5 or any(h.document_id != dataset.documents[0] for h in hits):
        raise ValueError("retrieval_scope_mismatch")
    if len({hit.chunk_id for hit in hits}) != len(hits):
        raise ValueError("duplicate_hit")
    return hashlib.sha256(
        json.dumps(
            [
                [h.ordinal, h.cosine_distance, hashlib.sha256(h.text.encode()).hexdigest()]
                for h in hits
            ],
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
