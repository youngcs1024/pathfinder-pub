"""Explicit, authorized E5.6 entry; never run full load through pytest collection."""

import argparse
from pathlib import Path

from tests.performance.capacity import run_suite


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["capacity", "queue", "retrieval-scale"])
    parser.add_argument(
        "--profile", required=True, choices=["capacity-e56-v1", "queue-e57-v1", "retrieval-e58-v1"]
    )
    parser.add_argument(
        "--authorization",
        required=True,
        choices=[
            "e56_local_capacity_user_approved_v1",
            "e57_local_queue_user_approved_v1",
            "e58_local_retrieval_user_approved_v1",
        ],
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        from tests.performance.queue import run_suite as run_queue
        from tests.performance.retrieval_scale import run_suite as run_retrieval

        execute = {"queue": run_queue, "capacity": run_suite, "retrieval-scale": run_retrieval}[
            args.command
        ]
        suite = execute(
            args.output, authorization=args.authorization, selected_profile=args.profile
        )
    except KeyboardInterrupt:
        print(f"{args.command}: cancelled")
        return 130
    except Exception:
        print(f"{args.command}: failed; inspect retained typed artifacts")
        return 1
    if args.command == "queue":
        from tests.performance.queue_contracts import successful_suite

        return 0 if successful_suite(suite) else 1
    return 0 if all(r.status == "PASS" for r in suite.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
