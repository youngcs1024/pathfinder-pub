"""Add Gate 6.5 action recovery terminal facts and lifecycle events."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014_gate6_action_recovery"
down_revision: str | None = "0013_gate6_mock_action_execution"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_EVENTS_BEFORE = (
    "'run.created', 'run.status_changed', 'run.completed', 'run.failed', "
    "'run.cancelled', 'job.lease_expired', 'job.dead', 'agent.plan.created', "
    "'agent.research.started', 'source.discovered', 'tool.started', "
    "'tool.finished', 'report.completed', 'rag.retrieved', 'action.proposed', "
    "'approval.expired', 'approval.decided', 'action.cancelled'"
)


def upgrade() -> None:
    op.drop_constraint(op.f("ck_tool_invocations_gate4_status"), "tool_invocations", type_="check")
    op.drop_constraint(
        op.f("ck_tool_invocations_terminal_fields"), "tool_invocations", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_tool_invocations_terminal_fields"),
        "tool_invocations",
        "(status = 'prepared' AND started_at IS NULL AND finished_at IS NULL "
        "AND latency_ms IS NULL AND result_summary IS NULL AND error_category IS NULL) OR "
        "(status = 'executing' AND started_at IS NOT NULL AND finished_at IS NULL "
        "AND latency_ms IS NULL AND result_summary IS NULL AND error_category IS NULL) OR "
        "(status = 'succeeded' AND started_at IS NOT NULL AND finished_at IS NOT NULL "
        "AND latency_ms IS NOT NULL AND result_summary IS NOT NULL AND error_category IS NULL) OR "
        "(status = 'failed' AND finished_at IS NOT NULL AND latency_ms IS NOT NULL "
        "AND result_summary IS NULL AND error_category IS NOT NULL) OR "
        "(status = 'outcome_unknown' AND started_at IS NOT NULL AND finished_at IS NOT NULL "
        "AND latency_ms IS NOT NULL AND result_summary IS NULL "
        "AND error_category = 'external_outcome_unknown')",
    )

    op.drop_constraint(op.f("ck_run_events_type"), "run_events", type_="check")
    op.create_check_constraint(
        op.f("ck_run_events_type"),
        "run_events",
        f"type IN ({_EVENTS_BEFORE}, 'action.started', 'action.completed', "
        "'action.failed', 'action.outcome_unknown')",
    )


def downgrade() -> None:
    bind = op.get_bind()
    unsafe = bind.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM tool_invocations WHERE status = 'outcome_unknown') "
            "OR EXISTS (SELECT 1 FROM run_events WHERE type IN "
            "('action.started', 'action.completed', 'action.failed', 'action.outcome_unknown'))"
        )
    ).scalar_one()
    if unsafe:
        raise RuntimeError("Gate 6.5 recovery data cannot be safely downgraded")

    op.drop_constraint(op.f("ck_run_events_type"), "run_events", type_="check")
    op.create_check_constraint(
        op.f("ck_run_events_type"), "run_events", f"type IN ({_EVENTS_BEFORE})"
    )
    op.drop_constraint(
        op.f("ck_tool_invocations_terminal_fields"), "tool_invocations", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_tool_invocations_terminal_fields"),
        "tool_invocations",
        "(status = 'prepared' AND started_at IS NULL AND finished_at IS NULL "
        "AND latency_ms IS NULL AND result_summary IS NULL AND error_category IS NULL) OR "
        "(status = 'executing' AND started_at IS NOT NULL AND finished_at IS NULL "
        "AND latency_ms IS NULL AND result_summary IS NULL AND error_category IS NULL) OR "
        "(status = 'succeeded' AND started_at IS NOT NULL AND finished_at IS NOT NULL "
        "AND latency_ms IS NOT NULL AND result_summary IS NOT NULL AND error_category IS NULL) OR "
        "(status = 'failed' AND finished_at IS NOT NULL AND latency_ms IS NOT NULL "
        "AND result_summary IS NULL AND error_category IS NOT NULL)",
    )
    op.create_check_constraint(
        op.f("ck_tool_invocations_gate4_status"),
        "tool_invocations",
        "status <> 'outcome_unknown'",
    )
