"""Explicit, authorized E5.6 entry; never run full load through pytest collection."""

import argparse
from pathlib import Path

from tests.performance.capacity import run_suite


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["capacity"])
    parser.add_argument("--profile", required=True, choices=["capacity-e56-v1"])
    parser.add_argument(
        "--authorization", required=True, choices=["e56_local_capacity_user_approved_v1"]
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        suite = run_suite(
            args.output, authorization=args.authorization, selected_profile=args.profile
        )
    except KeyboardInterrupt:
        print("capacity: cancelled")
        return 130
    except Exception:
        print("capacity: failed; inspect retained typed artifacts")
        return 1
    return 0 if all(r.status == "PASS" for r in suite.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
