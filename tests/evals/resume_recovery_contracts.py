"""Fixed D denominators and append-only deterministic recovery evidence."""

from collections import Counter
from dataclasses import asdict
from math import ceil

from app.worker.settings import WorkerRuntimeSettings
from tests.evals.product_acceptance_contracts import require

VERSION = "resume-recovery-d-v1"
FAULTS = ("after_claim", "before_publish", "in_transaction", "after_commit", "connection")
BLOCKS = ("revoked", "budget", "cancelled")
SETTINGS = asdict(WorkerRuntimeSettings())
SEED = 20260928
WINDOW_SECONDS = 120


def cases():
    result = []
    for mode in ("generation", "revision"):
        for fault in FAULTS:
            for ordinal in range(100):
                result.append(
                    dict(
                        case_id=f"{mode}-{fault}-{ordinal:03}",
                        mode=mode,
                        fault=fault,
                        kind="controlled" if ordinal < 90 else "real",
                        ordinal=ordinal,
                    )
                )
        for fault in BLOCKS:
            for ordinal in range(10):
                result.append(
                    dict(
                        case_id=f"{mode}-{fault}-{ordinal:03}",
                        mode=mode,
                        fault=fault,
                        kind="blocking",
                        ordinal=ordinal,
                    )
                )
    return result


def summarize(results, planned=None):
    planned = cases() if planned is None else planned
    expected = {c["case_id"]: c for c in planned}
    require(len(expected) == len(planned), "duplicate_planned_case")
    ids = [r["case"]["case_id"] for r in results]
    require(len(ids) == len(set(ids)) and set(ids) <= expected.keys(), "invalid_result_identity")
    for result in results:
        require(result["case"] == expected[result["case"]["case_id"]], "case_changed")
        require(result["status"] in ("PASS", "FAIL", "PARTIAL"), "invalid_case_status")
        if result["status"] == "PASS":
            require(result.get("verified") is True, "unverified_pass")
            require(result.get("injected") is True, "uninjected_pass")
            require(result.get("duplicate_side_effects", 0) == 0, "duplicate_publication_pass")
            if result["case"]["kind"] == "real":
                require(
                    result.get("clock") == "monotonic"
                    and 0 <= result["recovery_seconds"] <= WINDOW_SECONDS,
                    "invalid_real_timing",
                )
    groups = {}
    for case in planned:
        key = "/".join(case[k] for k in ("mode", "fault", "kind"))
        groups.setdefault(key, {"planned": 0, "pass": 0, "fail": 0, "partial": 0})["planned"] += 1
    for result in results:
        key = "/".join(result["case"][k] for k in ("mode", "fault", "kind"))
        groups[key][result["status"].lower()] += 1
    for group in groups.values():
        group["missing"] = group["planned"] - sum(group[k] for k in ("pass", "fail", "partial"))
        group["success_rate"] = group["pass"] / group["planned"]
    durations = sorted(
        r["recovery_seconds"]
        for r in results
        if r["case"]["kind"] == "real" and r.get("recovery_seconds") is not None
    )
    states = Counter(r["status"] for r in results)
    return {
        "version": VERSION,
        "status": "PASS"
        if len(results) == len(planned) and states["PASS"] == len(planned)
        else "PARTIAL"
        if len(results) < len(planned) or states["PARTIAL"]
        else "FAIL",
        "planned_recovery": sum(c["kind"] != "blocking" for c in planned),
        "planned_blocking": sum(c["kind"] == "blocking" for c in planned),
        "recorded": len(results),
        "groups": groups,
        "failed_cases": [r["case"]["case_id"] for r in results if r["status"] != "PASS"],
        "missing_cases": sorted(expected.keys() - set(ids)),
        "duplicate_side_effects": sum(r.get("duplicate_side_effects", 0) for r in results),
        "unverified_cases": sum(not r.get("verified", False) for r in results),
        "fake_model_attempts": sum(r.get("model_attempts", 0) for r in results),
        "replayed_model_attempts": sum(r.get("replayed_model_attempts", 0) for r in results),
        "real_latency": {
            "samples": len(durations),
            "p50_seconds": durations[ceil(len(durations) * 0.5) - 1] if durations else None,
            "p95_seconds": durations[ceil(len(durations) * 0.95) - 1] if durations else None,
            "max_seconds": max(durations, default=None),
            "timeouts": sum(
                r.get("timed_out", False) for r in results if r["case"]["kind"] == "real"
            ),
        },
        "review_kind": "DETERMINISTIC",
        "human_review": "NOT_RUN",
        "production_sla": "NOT_MEASURED",
    }
