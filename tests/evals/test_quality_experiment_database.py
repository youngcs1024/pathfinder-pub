"""Ownership and strict-reader boundaries without Docker, SQL or provider execution."""

import copy
import json
import os
import pickle
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tests.evals import quality_experiment_database as module
from tests.evals.quality_experiment_database import (
    BASE_GRAPHS,
    BASE_REVISION,
    GRAPH_CHECK,
    GRAPH_VERSION,
    ExperimentDatabaseError,
    ExperimentRun,
    OwnedDatabaseHandle,
    OwnedExperimentDatabase,
    require_owned_database,
    validate_graph_check,
)
from tests.evals.test_quality_e7a_contracts import context_for, report_for


def registered(monkeypatch):
    owner = OwnedExperimentDatabase()
    owner._engine = object()
    handle = object.__new__(OwnedDatabaseHandle)
    handle._owner, handle._pid = owner, os.getpid()
    owner._handle = handle
    monkeypatch.setitem(module._OWNERS, id(handle), owner)
    return owner, handle


@pytest.mark.parametrize(
    "value", [None, True, False, "postgresql://localhost/pathfinder_test", object()]
)
def test_arbitrary_database_confirmation_and_dsn_rejected(value):
    with pytest.raises(ExperimentDatabaseError, match=r"^invalid_handle$"):
        require_owned_database(value)


def test_constructor_and_unregistered_object_cannot_mint_handle():
    with pytest.raises(ExperimentDatabaseError, match="invalid_handle"):
        OwnedDatabaseHandle()
    with pytest.raises(ExperimentDatabaseError, match="invalid_handle"):
        require_owned_database(object.__new__(OwnedDatabaseHandle))


@pytest.mark.parametrize("change", ["pid", "owner", "closed", "engine", "identity"])
def test_stale_or_foreign_handle_rejected(monkeypatch, change):
    owner, handle = registered(monkeypatch)
    assert require_owned_database(handle) is owner
    if change == "pid":
        handle._pid += 1
    elif change == "owner":
        handle._owner = OwnedExperimentDatabase()
    elif change == "closed":
        owner._closed = True
    elif change == "engine":
        owner._engine = None
    else:
        owner._handle = object()
    with pytest.raises(ExperimentDatabaseError, match="invalid_handle"):
        require_owned_database(handle)


def test_handle_is_not_serializable_or_copyable_and_repr_has_no_identity(monkeypatch):
    owner, handle = registered(monkeypatch)
    assert owner._owner not in repr(handle) + repr(owner)
    for operation in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(ExperimentDatabaseError, match="invalid_handle"):
            operation(handle)


def test_owner_enforces_one_slot_and_exact_release(monkeypatch):
    owner, _ = registered(monkeypatch)
    one, two = object(), object()
    owner.claim_slot(one)
    for action in (lambda: owner.claim_slot(two), lambda: owner.release_slot(two)):
        with pytest.raises(ExperimentDatabaseError, match="resource_busy"):
            action()
    owner.release_slot(one)
    owner.claim_slot(two)
    owner.release_slot(two)


def wrapped(owner):
    return SimpleNamespace(
        id="owned-id",
        name=owner._name,
        reload=lambda: None,
        attrs={
            "Config": {"Labels": {module.LABEL: owner._owner}},
            "State": {"Running": True},
            "HostConfig": {
                "NetworkMode": "bridge",
                "NanoCpus": 2_000_000_000,
                "Memory": 2_147_483_648,
                "Binds": None,
                "Privileged": False,
            },
            "NetworkSettings": {
                "Ports": {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": "12345"}]}
            },
        },
    )


@pytest.mark.parametrize(
    "change", ["label", "id", "name", "ip", "port", "cpu", "memory", "bind", "network", "stopped"]
)
def test_container_identity_or_resource_drift_stops(monkeypatch, change):
    owner, _ = registered(monkeypatch)
    obj = wrapped(owner)
    owner._container_id = obj.id
    owner._container = SimpleNamespace(get_wrapped_container=lambda: obj)
    assert owner._verify_container() == 12345
    if change == "label":
        obj.attrs["Config"]["Labels"][module.LABEL] = "foreign"
    elif change == "id":
        obj.id = "foreign"
    elif change == "name":
        obj.name = "foreign"
    elif change in {"ip", "port"}:
        obj.attrs["NetworkSettings"]["Ports"]["5432/tcp"][0][
            "HostIp" if change == "ip" else "HostPort"
        ] = "0.0.0.0" if change == "ip" else "99999"
    elif change == "stopped":
        obj.attrs["State"]["Running"] = False
    else:
        field, value = {
            "cpu": ("NanoCpus", 1),
            "memory": ("Memory", 1),
            "bind": ("Binds", ["/private:/data"]),
            "network": ("NetworkMode", "host"),
        }[change]
        obj.attrs["HostConfig"][field] = value
    with pytest.raises(ExperimentDatabaseError):
        owner._verify_container()


def snapshot(versions=BASE_GRAPHS):
    expression = (
        "CHECK ((graph_version = ANY (ARRAY[" + ", ".join(f"'{v}'::text" for v in versions) + "])))"
    )
    return {"revision": (BASE_REVISION,), "constraints": [("runs", GRAPH_CHECK, expression, True)]}


@pytest.mark.parametrize(
    "change", ["version", "extra_graph", "missing_graph", "not_valid", "or_true", "missing"]
)
def test_schema_validator_rejects_drift(change):
    data = snapshot()
    if change == "version":
        data["revision"] = ("different",)
    elif change == "extra_graph":
        data = snapshot((*BASE_GRAPHS, GRAPH_VERSION))
    elif change == "missing_graph":
        data = snapshot(BASE_GRAPHS[:-1])
    elif change == "not_valid":
        data["constraints"][0] = (*data["constraints"][0][:3], False)
    elif change == "or_true":
        data["constraints"][0] = ("runs", GRAPH_CHECK, data["constraints"][0][2] + " OR true", True)
    else:
        data["constraints"] = []
    with pytest.raises(ExperimentDatabaseError, match="schema_drift"):
        validate_graph_check(data, BASE_GRAPHS)


def test_both_schema_versions_are_exact_and_extra_constraints_preserved():
    validate_graph_check(snapshot(), BASE_GRAPHS)
    validate_graph_check(snapshot((*BASE_GRAPHS, GRAPH_VERSION)), (*BASE_GRAPHS, GRAPH_VERSION))
    before = {**snapshot(), "columns": [("runs", "id")], "indexes": ["workspace"]}
    before["constraints"].append(("runs", "other", "CHECK (x > 0)", True))
    after = {
        **before,
        "constraints": [
            snapshot((*BASE_GRAPHS, GRAPH_VERSION))["constraints"][0],
            before["constraints"][1],
        ],
    }
    assert module._without_graph(before) == module._without_graph(after)


def run_context():
    context = context_for()
    run = ExperimentRun(
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        context.resume_document_id,
        GRAPH_VERSION,
        "application",
        context.request,
    )
    return run, context


@pytest.mark.parametrize(
    "change",
    ["unknown", "boolean", "reference", "contract", "nan", "duplicate", "context", "graph"],
)
def test_reader_rejects_invalid_candidate_json_or_identity(change):
    owner = OwnedExperimentDatabase()
    fixture, context = run_context()
    value = report_for(context, "sufficient", True).model_dump(mode="json")
    if change == "unknown":
        value["trusted_target"] = "secret-canary"
    elif change == "boolean":
        value["evidence_sufficient"] = False
    elif change == "reference":
        value["assessment"]["evidence_ids"] = ["unknown"]
    elif change == "contract":
        value["output_contract"] = "other"
    elif change == "context":
        context = replace(context, resume_document_id=uuid4())
    elif change == "graph":
        fixture = replace(fixture, graph_version="other")
    raw = json.dumps(value)
    if change == "nan":
        raw = raw[:-1] + ', "unknown": NaN}'
    if change == "duplicate":
        raw = raw[:-1] + ', "output_contract": "e7a_research_output_v1"}'
    with pytest.raises((ValueError, ExperimentDatabaseError)):
        owner._parse(fixture, raw, context)


async def test_cleanup_rejects_foreign_container_and_retains_failure(monkeypatch):
    owner, handle = registered(monkeypatch)
    owner._engine = None
    obj = wrapped(owner)
    obj.attrs["Config"]["Labels"][module.LABEL] = "foreign"
    stopped = []
    owner._container = SimpleNamespace(
        get_wrapped_container=lambda: obj, stop=lambda: stopped.append(1)
    )
    with pytest.raises(ExperimentDatabaseError, match="cleanup_failed"):
        await owner.aclose()
    assert owner.cleanup_failed and not stopped
    with pytest.raises(ExperimentDatabaseError, match="invalid_handle"):
        require_owned_database(handle)
