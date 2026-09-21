from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from tests.evals.product_acceptance_budget import admit, summarize
from tests.evals.product_acceptance_contracts import (
    AcceptanceError,
    Budget,
    Manifest,
    new_directory,
    publish,
    read_private_json,
    review_digest,
    validate_review,
    verify_manifest,
    write_decision,
)
from tests.evals.product_acceptance_environment import (
    LABEL,
    OwnedProductDatabase,
    local_docker,
    validate_owned,
)
from tests.evals.product_acceptance_runtime import accepted_facts
from tests.evals.quality_experiment_binding import ExperimentError


def review_fixture():
    digest = "a" * 64
    return {
        "action_intent_id": str(uuid4()),
        "action_key": "submit_application",
        "action_revision": 1,
        "tool_name": "submit_mock_application",
        "effect": "irreversible",
        "args_snapshot": {"text": "synthetic exact draft"},
        "canonicalization_version": 1,
        "args_digest": digest,
        "target_snapshot": {"portal": "internal_mock"},
        "target_canonicalization_version": 1,
        "target_digest": digest,
        "approval_binding_version": 1,
        "approval_binding_digest": digest,
        "status": "proposed",
        "recovery_attempts": 0,
        "result": None,
        "evidence": None,
        "manual_review_required": False,
        "manual_review_instruction": None,
        "decision": None,
        "approval_request": {
            "request_id": str(uuid4()),
            "status": "pending",
            "version": 1,
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "args_digest": digest,
            "target_digest": digest,
            "approval_binding_version": 1,
            "approval_binding_digest": digest,
            "policy_version": 1,
            "policy_snapshot": {},
        },
    }


def private_root(tmp_path):
    root = tmp_path / "acceptance"
    root.mkdir(mode=0o700)
    return root


def test_private_review_exact_digest_and_create_only(tmp_path):
    root = private_root(tmp_path)
    review = review_fixture()
    publish(root / "mixed_alpha-review.json", review)
    write_decision(root, "mixed_alpha", "approve", review_digest(review))
    decision = read_private_json(root / "mixed_alpha-decision.json")
    assert decision["review_digest"] == review_digest(review)
    assert decision["reviewed_exact_content"] is True
    with pytest.raises(ExperimentError):
        write_decision(root, "mixed_alpha", "approve", review_digest(review))
    assert (root / "mixed_alpha-review.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("change", ["args", "target", "request", "version", "expiry", "policy"])
def test_every_exact_review_change_invalidates_decision(change):
    original = review_fixture()
    updated = json.loads(json.dumps(original))
    if change == "args":
        updated["args_snapshot"]["text"] = "different"
    elif change == "target":
        updated["target_snapshot"]["portal"] = "different"
    elif change == "request":
        updated["approval_request"]["request_id"] = str(uuid4())
    elif change == "version":
        updated["approval_request"]["version"] += 1
    elif change == "expiry":
        updated["approval_request"]["expires_at"] = datetime.now(UTC).isoformat()
    else:
        updated["approval_request"]["policy_snapshot"]["changed"] = True
    assert review_digest(original) != review_digest(updated)


@pytest.mark.parametrize(
    "change,category",
    [
        ("expiry", "approval_expired"),
        ("status", "approval_not_pending"),
        ("binding", "approval_binding_mismatch"),
    ],
)
def test_invalid_approval_is_refused(change, category):
    review = review_fixture()
    if change == "expiry":
        review["approval_request"]["expires_at"] = datetime.now(UTC).isoformat()
    elif change == "status":
        review["approval_request"]["status"] = "consumed"
    else:
        review["approval_request"]["args_digest"] = "b" * 64
    with pytest.raises(AcceptanceError, match=category):
        validate_review(review)


def test_missing_review_wrong_digest_and_closed_execution_refused(tmp_path):
    root = private_root(tmp_path)
    review = review_fixture()
    with pytest.raises(FileNotFoundError):
        write_decision(root, "mixed_alpha", "approve", review_digest(review))
    publish(root / "mixed_alpha-review.json", review)
    with pytest.raises(AcceptanceError, match="review_changed"):
        write_decision(root, "mixed_alpha", "approve", "sha256:" + "0" * 64)
    publish(root / "report.json", {})
    with pytest.raises(AcceptanceError, match="execution_closed"):
        write_decision(root, "mixed_alpha", "approve", review_digest(review))


def test_secret_and_link_artifacts_fail_without_writing(tmp_path):
    root = private_root(tmp_path)
    path = root / "private.json"
    with pytest.raises(AcceptanceError, match="unsafe_artifact"):
        publish(path, {"text": "private-key-marker"}, ("private-key-marker",))
    assert not path.exists()
    link = tmp_path / "linked"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ExperimentError):
        publish(link / "private.json", {})
    with pytest.raises(AcceptanceError):
        new_directory(root)


def usage(**changes):
    return {**summarize([], provider="qwen"), **changes}


@pytest.mark.parametrize(
    "changes,category",
    [
        ({"unfinished": 1}, "unfinished_accounting"),
        ({"unknown_cost": 1}, "unknown_usage_or_cost"),
        ({"unknown_usage": 1}, "unknown_usage_or_cost"),
        ({"attempts": 100}, "budget_exhausted"),
        ({"input_tokens": 500000}, "budget_exhausted"),
        ({"output_tokens": 200000}, "budget_exhausted"),
        ({"known_cost_cny": "4.91"}, "budget_exhausted"),
    ],
)
def test_budget_refuses_next_attempt(changes, category):
    with pytest.raises(AcceptanceError, match=category):
        admit(usage(**changes), Budget())


def test_inflight_boundary_is_honest_and_unknown_never_zero():
    admit(usage(known_cost_cny="4.90"), Budget())
    admit(usage(attempts=100, known_cost_cny="5.00"), Budget(), after=True)
    with pytest.raises(AcceptanceError, match="budget_exhausted"):
        admit(usage(known_cost_cny="5.01"), Budget(), after=True)
    rows = [NS(status="failed", token_usage=None, estimated_cost=None)]
    result = summarize(rows, provider="qwen")
    assert result["unknown_cost"] == result["unknown_usage"] == 1
    assert result["attempts"] == 1
    rows.append(
        NS(
            status="succeeded",
            token_usage={"input_tokens": 7, "output_tokens": 3},
            estimated_cost=Decimal("0.20"),
        )
    )
    assert summarize(rows, provider="qwen")["known_cost_cny"] == "0.20"


def container_fixture():
    return {
        "Id": "owned",
        "Name": "/pf-e83-owner",
        "Config": {"Labels": {LABEL: "owner"}},
        "HostConfig": {
            "NetworkMode": "bridge",
            "NanoCpus": 2000000000,
            "Memory": 2147483648,
            "PidsLimit": 128,
        },
        "State": {"Running": True},
        "NetworkSettings": {"Ports": {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": "54321"}]}},
    }


@pytest.mark.parametrize("change", ["owner", "id", "name", "memory", "network", "port", "bind"])
def test_database_identity_and_boundaries(change):
    info = container_fixture()
    assert validate_owned(info, owner="owner", container_id="owned", running=True) == 54321
    if change == "owner":
        info["Config"]["Labels"][LABEL] = "foreign"
    elif change == "id":
        info["Id"] = "foreign"
    elif change == "name":
        info["Name"] = "/foreign"
    elif change == "memory":
        info["HostConfig"]["Memory"] = 0
    elif change == "network":
        info["HostConfig"]["NetworkMode"] = "host"
    elif change == "bind":
        info["HostConfig"]["Binds"] = ["/host:/container"]
    else:
        info["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostIp"] = "0.0.0.0"
    with pytest.raises(AcceptanceError):
        validate_owned(info, owner="owner", container_id="owned", running=True)


def test_cleanup_refuses_foreign_container_and_preserves_evidence(tmp_path, monkeypatch):
    root = private_root(tmp_path)
    db = OwnedProductDatabase(root, "image")
    db.container_id = "owned"
    calls = []
    monkeypatch.setattr(
        db, "inspect", lambda **kwargs: (_ for _ in ()).throw(AcceptanceError("ownership_mismatch"))
    )
    monkeypatch.setattr(
        "tests.evals.product_acceptance_environment.docker", lambda *a: calls.append(a)
    )
    with pytest.raises(AcceptanceError):
        db.stop()
    assert calls == []
    assert read_private_json(root / "cleanup.json")["stopped"] is False


def test_remote_docker_rejected_before_inspection(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.invalid:2375")
    with pytest.raises(AcceptanceError, match="remote_docker_rejected"):
        local_docker()


def test_rejection_is_not_confused_with_failed_run_or_model_quality():
    facts = {
        "run_status": "completed",
        "job_statuses": ["done"],
        "graph_version": "pathfinder-research-v6",
        "checkpoint_count": 5,
        "decisions": ["reject"],
        "action_statuses": ["cancelled"],
        "request_statuses": ["rejected"],
        "mock_count": 0,
        "web_calls": 1,
        "retrieval_calls": 1,
        "rag_events": 1,
    }
    assert accepted_facts(facts, "reject")
    assert not accepted_facts({**facts, "mock_count": 1}, "reject")
    assert not accepted_facts({**facts, "retrieval_calls": 0}, "reject")


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--root", "/tmp/unused", "--credentials", "/tmp/unused"],
        [
            "decide",
            "--root",
            "/tmp/unused",
            "--case",
            "mixed_alpha",
            "--decision",
            "approve",
            "--review-digest",
            "sha256:" + "0" * 64,
        ],
    ],
)
def test_live_and_decision_require_explicit_flags(arguments):
    from tests.evals.product_acceptance import main

    with pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == 2


def manifest_fixture():
    return Manifest(
        source_sha="a" * 40, source_tree="b" * 40, files={}, dataset_digest="sha256:" + "c" * 64
    )


@pytest.mark.parametrize(
    "field,value,category",
    [
        ("budget", Budget(cost_admission_budget_cny=Decimal("6")), "budget_changed"),
        ("chat_model", "changed-model", "version_changed"),
        ("pricing_version", "changed-price", "version_changed"),
    ],
)
def test_manifest_cannot_expand_authorization(field, value, category):
    manifest = manifest_fixture().model_copy(update={field: value})
    with pytest.raises(AcceptanceError, match=category):
        verify_manifest(manifest)


def test_changed_source_files_refused_before_execution(monkeypatch):
    import tests.evals.product_acceptance_contracts as contracts

    manifest = manifest_fixture()
    monkeypatch.setattr(contracts, "source_identity", lambda *a: manifest.source_tree)
    monkeypatch.setattr(contracts, "bound_files", lambda: ["new-source"])
    with pytest.raises(AcceptanceError, match="source_changed"):
        verify_manifest(manifest)


async def test_setup_failure_keeps_all_unexecuted_slots_and_safe_error(tmp_path, monkeypatch):
    from tests.evals import product_acceptance as cli

    root = private_root(tmp_path)
    marker = "sensitive-exception-canary"

    def missing(*args):
        raise ValueError(marker)

    monkeypatch.setattr(cli, "credentials", missing)
    with pytest.raises(AcceptanceError, match="execution_failed"):
        await cli._execute_once(root, root / "missing", manifest_fixture(), {"image_id": "image"})
    report = read_private_json(root / "report.json")
    assert report["status"] == "PARTIAL"
    assert [case["status"] for case in report["cases"]] == ["NOT_RUN", "NOT_RUN"]
    assert marker not in (root / "report.json").read_text()


def test_cli_does_not_print_exception_body(tmp_path, monkeypatch, capsys):
    from tests.evals import product_acceptance as cli

    def failed(*args):
        raise ValueError("private-business-exception-canary")

    monkeypatch.setattr(cli, "prepare", failed)
    assert cli.main(["prepare", "--root", str(tmp_path / "unused")]) == 1
    assert "private-business-exception-canary" not in capsys.readouterr().out
