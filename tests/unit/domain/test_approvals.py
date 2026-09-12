from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.approvals import (
    ApprovalStatus,
    fixed_approval_policy,
    is_valid_approval_transition,
    persisted_approval_status,
    validate_approval_transition,
)
from app.domain.errors import DomainConflictError, DomainInvariantError

_APPROVAL_TRANSITIONS = {
    ApprovalStatus.PENDING: {
        ApprovalStatus.APPROVED,
        ApprovalStatus.REJECTED,
        ApprovalStatus.EXPIRED,
    },
    ApprovalStatus.APPROVED: {ApprovalStatus.CONSUMED, ApprovalStatus.EXPIRED},
    ApprovalStatus.REJECTED: set(),
    ApprovalStatus.CONSUMED: set(),
    ApprovalStatus.EXPIRED: set(),
}


@pytest.mark.parametrize("current", tuple(ApprovalStatus))
@pytest.mark.parametrize("target", tuple(ApprovalStatus))
def test_approval_transition_matrix(current: ApprovalStatus, target: ApprovalStatus) -> None:
    expected = target in _APPROVAL_TRANSITIONS[current]

    assert is_valid_approval_transition(current, target) is expected
    if expected:
        validate_approval_transition(current, target)
    else:
        with pytest.raises(DomainConflictError):
            validate_approval_transition(current, target)


def test_persisted_unknown_approval_state_is_an_invariant_failure() -> None:
    with pytest.raises(DomainInvariantError):
        persisted_approval_status("cancelled")


def test_fixed_mvp_policy_snapshot_is_exact_and_immutable() -> None:
    policy = fixed_approval_policy()

    assert policy.model_dump(mode="json") == {
        "eligible_roles": ["reviewer", "admin"],
        "required_approvals": 1,
        "separation_of_duty": False,
    }
    with pytest.raises(ValidationError):
        policy.required_approvals = 2  # type: ignore[misc]
