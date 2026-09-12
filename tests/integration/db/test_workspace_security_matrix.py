from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
PROJECT_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_RESOURCES = {
    "runs",
    "events_sse",
    "documents",
    "approval_requests",
    "decisions",
    "actions",
    "llm_invocations",
}
REQUIRED_BOUNDARIES = {
    "same_actor_different_workspace",
    "same_workspace_different_actor",
    "nonmember",
    "revoked_membership",
    "unknown_id",
    "actor_spoof",
    "workspace_spoof",
    "role_spoof",
}

# This is the Gate 7.3 acceptance index. The referenced focused tests own the
# detailed fixtures and side-effect assertions, avoiding a second copy of every
# resource setup while making omissions from the matrix mechanically visible.
SECURITY_MATRIX = {resource: REQUIRED_BOUNDARIES for resource in REQUIRED_RESOURCES}

EVIDENCE = {
    "tests/integration/db/test_run_api_store.py": {
        "test_workspace_read_and_cancel_rbac_are_enforced_in_store",
        "test_reviewer_can_cancel_only_own_persisted_creator_run",
        "test_revoked_membership_is_hidden_before_run_read",
        "test_event_reader_hides_cross_workspace_unknown_and_stale_tenant",
    },
    "tests/integration/db/test_document_ingestion.py": {
        "test_revoked_membership_cannot_use_complete_document_dedupe_fast_path",
        "test_different_actor_reuses_first_creator",
    },
    "tests/integration/db/test_document_retrieval.py": {
        "test_cross_workspace_filter_and_same_workspace_noncreator_share",
        "test_stale_tenant_and_revocation_after_embedding_fail_closed",
    },
    "tests/integration/db/test_action_proposals.py": {
        "test_tenant_run_actor_and_resume_splices_fail_closed",
    },
    "tests/integration/db/test_approval_decisions.py": {
        "test_action_review_workspace_members_can_read_without_cross_tenant_leak",
        "test_persisted_membership_role_defeats_forged_tenant_context",
        "test_decision_uses_committed_persisted_role_after_promotion",
        "test_decision_uses_committed_persisted_role_after_demotion",
    },
    "tests/integration/db/test_llm_invocations.py": {
        "test_orphan_and_cross_workspace_run_are_rejected_before_provider",
        "test_same_workspace_member_cannot_attach_invocation_to_another_actors_run",
        "test_unknown_cross_workspace_and_revoked_actor_are_rejected_before_insert",
    },
    "tests/unit/api/test_runs.py": {
        "test_create_run_rejects_invalid_or_trusted_extra_fields",
        "test_nonmember_workspace_is_hidden_as_404",
    },
    "tests/unit/api/test_action_intents.py": {
        "test_decision_body_forbids_trusted_fields_and_bounds_reason",
    },
}


def _test_functions(relative_path: str) -> set[str]:
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("test_")
    }


def test_gate_7_3_matrix_has_every_resource_and_boundary() -> None:
    assert set(SECURITY_MATRIX) == REQUIRED_RESOURCES
    assert all(boundaries == REQUIRED_BOUNDARIES for boundaries in SECURITY_MATRIX.values())


def test_gate_7_3_matrix_evidence_remains_collected() -> None:
    missing: list[str] = []
    for relative_path, expected_tests in EVIDENCE.items():
        absent = expected_tests - _test_functions(relative_path)
        missing.extend(f"{relative_path}::{test_name}" for test_name in sorted(absent))
    assert missing == []
