"""Explicit, authorized benchmarks; ordinary CI only runs bounded test cases."""

import argparse
import asyncio
from pathlib import Path

COMMANDS = {
    "smoke": ("instant-v1", "e510_local_smoke_user_approved_v1"),
    "capacity": ("capacity-e56-v1", "e56_local_capacity_user_approved_v1"),
    "queue": ("queue-e57-v1", "e57_local_queue_user_approved_v1"),
    "retrieval-scale": ("retrieval-e58-v1", "e58_local_retrieval_user_approved_v1"),
    "faults": ("faults-e59-v1", "e59_local_faults_user_approved_v1"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Owned loopback benchmarks, always fake/fake/fake/off.",
        epilog=(
            "Smoke: one Run, one worker, at most 128 HTTP requests / 64 calls / "
            "40 business seconds. Full capacity, queue, retrieval and fault suites "
            "require manual authorization and are not ordinary CI workloads. "
            "Output must be a new absolute directory outside the repository under "
            "an owned private parent. Retain partial artifacts; never overwrite or delete them. "
            "Exit: 0 complete, 1 execution/evidence/cleanup failure, 2 invalid arguments, "
            "130 interrupted. GNU Make reports recipe failure as 2. See make benchmark-help."
        ),
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--profile",
        required=True,
        choices=[value[0] for value in COMMANDS.values()],
    )
    parser.add_argument(
        "--authorization",
        required=True,
        choices=[value[1] for value in COMMANDS.values()],
    )
    parser.add_argument(
        "--mode", choices=["research", "application"], help="smoke only; application by default"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if (args.profile, args.authorization) != COMMANDS[args.command]:
        parser.error("profile and authorization must match the command")
    if args.command != "smoke" and args.mode is not None:
        parser.error("mode is only valid for smoke")
    if not args.output.is_absolute():
        parser.error("output must be an absolute path")
    try:
        if args.command == "smoke":
            from tests.performance.smoke import run_smoke

            result = run_smoke(args.output, mode=args.mode or "application")
            print(f"smoke: {result.status}; {result.category or 'complete'}")
            return 0 if result.status == "PASS" else 1

        from tests.performance.capacity import run_suite
        from tests.performance.faults import run_suite as run_faults
        from tests.performance.queue import run_suite as run_queue
        from tests.performance.retrieval_scale import run_suite as run_retrieval

        execute = {
            "queue": run_queue,
            "capacity": run_suite,
            "retrieval-scale": run_retrieval,
            "faults": run_faults,
        }[args.command]
        suite = execute(
            args.output, authorization=args.authorization, selected_profile=args.profile
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"{args.command}: cancelled")
        return 130
    except Exception:
        print(f"{args.command}: failed; inspect retained typed artifacts")
        return 1
    if args.command == "queue":
        from tests.performance.queue_contracts import successful_suite

        return 0 if successful_suite(suite) else 1
    return 0 if suite.results and all(r.status == "PASS" for r in suite.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
