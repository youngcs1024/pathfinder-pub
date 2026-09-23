from enum import StrEnum
from itertools import product
from uuid import uuid4

import pytest

from app.domain.jobs import JobStatus, is_valid_job_transition
from app.domain.provisioning import WorkspaceRole
from app.domain.runs import (
    CURRENT_GRAPH_VERSION,
    SUPPORTED_GRAPH_VERSIONS,
    MessageRole,
    RunMode,
    RunStatus,
    is_valid_run_transition,
)
from app.domain.tenancy import TenantContext
from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import (
    ToolInvocationStatus,
    is_valid_tool_invocation_transition,
)
from app.events.contracts import CURRENT_RUN_EVENT_VERSION, RunEventType


def _value_set(enum_type: type[StrEnum]) -> set[str]:
    return {item.value for item in enum_type}


def test_persistent_state_enum_values_are_exact() -> None:
    assert _value_set(MessageRole) == {"user", "assistant"}
    assert _value_set(RunMode) == {
        "research",
        "application",
        "material_preparation",
        "resume_generation",
        "resume_revision",
    }
    assert _value_set(RunStatus) == {
        "queued",
        "running",
        "waiting_approval",
        "completed",
        "failed",
        "cancelled",
    }
    assert _value_set(JobStatus) == {"queued", "leased", "done", "dead"}
    assert _value_set(ToolInvocationStatus) == {
        "prepared",
        "executing",
        "succeeded",
        "failed",
        "outcome_unknown",
    }
    assert _value_set(ToolEffect) == {
        "read_only",
        "reversible",
        "irreversible",
    }


def test_graph_and_event_versions_are_fixed() -> None:
    assert CURRENT_GRAPH_VERSION == "pathfinder-research-v6"
    assert SUPPORTED_GRAPH_VERSIONS == frozenset({"pathfinder-resume-v1"})
    assert CURRENT_RUN_EVENT_VERSION == 1
    assert _value_set(RunEventType) == {
        "run.created",
        "run.status_changed",
        "run.completed",
        "run.failed",
        "run.cancelled",
        "job.lease_expired",
        "job.dead",
        "agent.plan.created",
        "agent.research.started",
        "source.discovered",
        "tool.started",
        "tool.finished",
        "report.completed",
        "rag.retrieved",
        "action.proposed",
        "approval.expired",
        "approval.decided",
        "action.cancelled",
        "action.started",
        "action.completed",
        "action.failed",
        "action.outcome_unknown",
    }


@pytest.mark.parametrize(
    ("current", "target"),
    list(product(RunStatus, repeat=2)),
)
def test_run_transition_matrix_is_exhaustive(
    current: RunStatus,
    target: RunStatus,
) -> None:
    allowed = {
        (RunStatus.QUEUED, RunStatus.RUNNING),
        (RunStatus.QUEUED, RunStatus.CANCELLED),
        (RunStatus.RUNNING, RunStatus.WAITING_APPROVAL),
        (RunStatus.RUNNING, RunStatus.COMPLETED),
        (RunStatus.RUNNING, RunStatus.FAILED),
        (RunStatus.RUNNING, RunStatus.CANCELLED),
        (RunStatus.WAITING_APPROVAL, RunStatus.RUNNING),
        (RunStatus.WAITING_APPROVAL, RunStatus.CANCELLED),
    }
    assert is_valid_run_transition(current, target) is ((current, target) in allowed)


@pytest.mark.parametrize(
    ("current", "target"),
    list(product(JobStatus, repeat=2)),
)
def test_job_transition_matrix_is_exhaustive(
    current: JobStatus,
    target: JobStatus,
) -> None:
    allowed = {
        (JobStatus.QUEUED, JobStatus.LEASED),
        (JobStatus.QUEUED, JobStatus.DONE),
        (JobStatus.LEASED, JobStatus.DONE),
        (JobStatus.LEASED, JobStatus.QUEUED),
        (JobStatus.LEASED, JobStatus.DEAD),
        (JobStatus.DONE, JobStatus.QUEUED),
    }
    assert is_valid_job_transition(current, target) is ((current, target) in allowed)


@pytest.mark.parametrize(
    ("current", "target"),
    list(product(ToolInvocationStatus, repeat=2)),
)
def test_tool_transition_matrix_is_exhaustive(
    current: ToolInvocationStatus,
    target: ToolInvocationStatus,
) -> None:
    allowed = {
        (ToolInvocationStatus.PREPARED, ToolInvocationStatus.EXECUTING),
        (ToolInvocationStatus.PREPARED, ToolInvocationStatus.FAILED),
        (ToolInvocationStatus.EXECUTING, ToolInvocationStatus.SUCCEEDED),
        (ToolInvocationStatus.EXECUTING, ToolInvocationStatus.FAILED),
        (ToolInvocationStatus.EXECUTING, ToolInvocationStatus.OUTCOME_UNKNOWN),
    }
    assert is_valid_tool_invocation_transition(current, target) is ((current, target) in allowed)


def test_tenant_context_is_typed_frozen_and_repr_safe() -> None:
    workspace_id = uuid4()
    actor_user_id = uuid4()
    context = TenantContext(
        workspace_id=workspace_id,
        actor_user_id=actor_user_id,
        role=WorkspaceRole.REVIEWER,
    )

    assert context.workspace_id == workspace_id
    assert context.actor_user_id == actor_user_id
    assert context.role is WorkspaceRole.REVIEWER
    assert repr(context).startswith("<app.domain.tenancy.TenantContext object at ")

    with pytest.raises(AttributeError):
        context.role = WorkspaceRole.ADMIN  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("workspace_id", "not-a-uuid", "tenant workspace_id must be a UUID"),
        ("actor_user_id", "not-a-uuid", "tenant actor_user_id must be a UUID"),
        ("role", "admin", "tenant role must be a WorkspaceRole"),
    ],
)
def test_tenant_context_rejects_untyped_values(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "workspace_id": uuid4(),
        "actor_user_id": uuid4(),
        "role": WorkspaceRole.ADMIN,
    }
    values[field] = value

    with pytest.raises(TypeError, match=message):
        TenantContext(**values)  # type: ignore[arg-type]
