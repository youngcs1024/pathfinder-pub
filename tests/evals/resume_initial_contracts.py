"""Fixed A denominators and equal-JD summaries, independent of observed success."""

from decimal import Decimal
from statistics import mean

from tests.evals.product_acceptance_contracts import require
from tests.evals.resume_experiment_scoring import ratio

VERSION = "resume-initial-a-v2"
ARMS = ("one_shot", "selection", "pathfinder")


def samples(case_ids, *, pilot=False):
    result = []
    for repeat in range(1, 2 if pilot else 4):
        for ordinal, case_id in enumerate(case_ids):
            offset = (ordinal + repeat - 1) % 3
            for arm in (*ARMS[offset:], *ARMS[:offset]):
                result.append(
                    {
                        "sample_id": f"{case_id}-r{repeat}-{arm}",
                        "case_id": case_id,
                        "repeat": repeat,
                        "arm": arm,
                        "c_start": not pilot and repeat == 1 and arm != "selection",
                    }
                )
    return result


def summarize(planned, results, annotations):
    ids = [s["sample_id"] for s in planned]
    require(len(ids) == len(set(ids)) and set(results) <= set(ids), "invalid_sample_set")
    groups = {}
    for arm in ARMS:
        cases = {}
        for case_id in dict.fromkeys(s["case_id"] for s in planned):
            chosen = [s for s in planned if s["case_id"] == case_id and s["arm"] == arm]
            rows = [results.get(s["sample_id"], {}) for s in chosen]
            applicable = sum(r["applicable"] for r in annotations[case_id]["requirements"])
            metrics = {}
            for metric in (
                "fact_support",
                "profile_fact_support",
                "coverage",
                "condition_omission",
            ):
                vals, na, unresolved = [], 0, 0
                for row in rows:
                    measured = (row.get("metrics") or {}).get(metric)
                    if measured is not None:
                        if measured["value"] is None:
                            na += 1
                        else:
                            vals.append(measured["value"])
                    elif row.get("generation_status") == "FAILED":
                        if metric == "coverage" and applicable:
                            vals.append(0.0)
                        else:
                            na += 1
                    else:
                        unresolved += 1
                metrics[metric] = {
                    "mean": mean(vals) if vals else None,
                    "assessed": len(vals),
                    "not_applicable": na,
                    "unresolved_or_missing": unresolved,
                }
            cases[case_id] = {
                "planned": len(chosen),
                "recorded": sum(bool(r) for r in rows),
                "generation_failed": sum(r.get("generation_status") == "FAILED" for r in rows),
                "scoring_unresolved": sum(r.get("score_status") == "UNRESOLVED" for r in rows),
                "usable": ratio(
                    sum((r.get("metrics") or {}).get("usable", False) for r in rows), len(chosen)
                ),
                "metrics": metrics,
            }
        arm_rows = [
            results[s["sample_id"]]
            for s in planned
            if s["arm"] == arm and s["sample_id"] in results
        ]
        groups[arm] = {"by_jd": cases, "cross_jd": {}, "cost": {}}
        for stage, key in (
            ("generation", "usage"),
            ("scoring", "score_usage"),
            ("initial", "initial"),
            ("correction", "correction"),
            ("review", "review"),
        ):
            usages = (
                [r[key] for r in arm_rows if key in r]
                if stage in ("generation", "scoring")
                else [r["score_stages"][key] for r in arm_rows if key in r.get("score_stages", {})]
            )
            groups[arm]["cost"][stage] = {
                **{
                    k: sum(u.get(k, 0) for u in usages)
                    for k in (
                        "attempts",
                        "input_tokens",
                        "output_tokens",
                        "latency_ms",
                        "unknown_cost",
                        "unknown_usage",
                    )
                },
                "known_cost_cny": str(
                    sum((Decimal(u["known_cost_cny"]) for u in usages), Decimal(0))
                ),
                "cost_status": "ESTIMATED",
                "includes_failed_samples": True,
            }
        groups[arm]["generation_elapsed_seconds"] = sum(
            r.get("elapsed_seconds", 0) for r in arm_rows
        )
        for metric in ("fact_support", "profile_fact_support", "coverage", "condition_omission"):
            vals = [
                v["metrics"][metric]["mean"]
                for v in cases.values()
                if v["metrics"][metric]["mean"] is not None
            ]
            unresolved = sum(v["metrics"][metric]["unresolved_or_missing"] for v in cases.values())
            groups[arm]["cross_jd"][metric] = {
                "mean": mean(vals) if vals else None,
                "jd_denominator": len(vals),
                "jd_total": len(cases),
                "unresolved_or_missing_samples": unresolved,
                "complete": unresolved == 0,
            }
        groups[arm]["usable"] = ratio(
            sum(v["usable"]["numerator"] for v in cases.values()),
            sum(v["planned"] for v in cases.values()),
        )
    missing = len(ids) - len(results)
    unresolved = sum(r.get("score_status") == "UNRESOLVED" for r in results.values())
    return {
        "version": VERSION,
        "planned": len(ids),
        "recorded": len(results),
        "missing": missing,
        "unresolved": unresolved,
        "groups": groups,
        "status": "PASS" if not missing and not unresolved else "PARTIAL",
        "review_kind": "AGENT_ASSESSED",
        "human_review": "NOT_RUN",
        "costs_include_failed": True,
        "zero_fully_supported_jds": [
            k
            for k, v in annotations.items()
            if not any(r["applicable"] and r["support"] == "full" for r in v["requirements"])
        ],
    }
