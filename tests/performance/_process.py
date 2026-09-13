"""Fixed child roles; production assembly with test-only socket/readiness adaptation."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from tests.performance.environment import ROOT, EnvironmentError, send_packet


def main() -> None:
    bootstrap = json.loads(sys.stdin.buffer.read(8193))
    owner = UUID(bootstrap["owner"]).hex
    role = sys.argv[1]
    if any(
        os.environ.get(key) != value
        for key, value in {
            "PF_LLM_MODE": "fake",
            "PF_SEARCH_MODE": "fake",
            "PF_AUTH_MODE": "fake",
            "PF_TRACE_MODE": "off",
        }.items()
    ):
        raise EnvironmentError("invalid_profile")
    from app.config import Settings

    settings = Settings()
    # Child receives only the supervisor-generated target. Reject accidental fallback again.
    from sqlalchemy import make_url

    url = make_url(settings.database_url.get_secret_value())
    if (
        url.host != "127.0.0.1"
        or url.database != "pf_e52_" + owner
        or url.username != "pf_e52"
        or not url.password
        or url.query
        or url.drivername != "postgresql+psycopg"
        or not url.port
    ):
        raise EnvironmentError("ownership_mismatch")
    if type(bootstrap.get("metrics", False)) is not bool:
        raise EnvironmentError("invalid_profile")
    from tests.performance.metrics import Collector
    from tests.performance.metrics_runtime import instrument, instrument_application

    if bootstrap.get("queue") is True:
        from tests.performance.queue_metrics import QueueCollector

        collector = QueueCollector(role, Path(bootstrap["output_dir"]))
    elif bootstrap.get("capacity") is True:
        from tests.performance.capacity_metrics import CapacityCollector

        collector = CapacityCollector(role, Path(bootstrap["output_dir"]))
    else:
        collector = (
            Collector(role, Path(bootstrap["output_dir"])) if bootstrap.get("metrics") else None
        )
    if role == "migrate":
        from alembic import command
        from alembic.config import Config

        config = Config(str(ROOT / "alembic.ini"))
        config.attributes["database_url"] = settings.database_url.get_secret_value()
        command.upgrade(config, "head")
    elif role == "api":
        import uvicorn

        from app import main as api

        application = api.create_app(settings)
        if collector is not None:
            instrument_application(application, api, collector)
        with socket.socket(fileno=bootstrap["socket_fd"]) as listener:
            if listener.getsockname()[0] != "127.0.0.1":
                raise EnvironmentError("unsafe_binding")
            server = uvicorn.Server(
                uvicorn.Config(
                    application,
                    lifespan="on",
                    access_log=False,
                    log_config=None,
                    workers=1,
                )
            )
            asyncio.run(server.serve(sockets=[listener]))
    elif role == "worker":
        from app.worker import main as worker

        @contextmanager
        def readiness(_path: Path):
            with socket.socket(fileno=bootstrap["ready_fd"]) as channel:
                send_packet(channel, {"owner": owner, "kind": "worker_ready"})
            yield

        # Adapt only the operational readiness marker in this child. The production assembly,
        # lifecycle, runner, graph, repositories and factory remain the actual implementations.
        adapters = nullcontext()
        if "call_profile" in bootstrap:
            from tests.performance.adapters import Calls, worker_adapters
            from tests.performance.workload import parse_profile

            calls = Calls(
                parse_profile(bootstrap["call_profile"]), directory=Path(bootstrap["output_dir"])
            )
            if "fault_config" in bootstrap:
                from tests.performance.fault_runtime import FaultCalls

                calls = FaultCalls(bootstrap)
            adapters = worker_adapters(worker, calls)
        faults = nullcontext()
        if "fault_config" in bootstrap:
            from tests.performance.fault_runtime import instrument_faults

            faults = instrument_faults(worker, calls)
        with (
            patch.object(worker, "_clear_worker_ready_marker", lambda _path: None),
            patch.object(worker, "_worker_readiness_marker", readiness),
            adapters,
            faults,
            instrument(worker, collector) if collector is not None else nullcontext(),
        ):
            asyncio.run(worker.run_worker(settings))
    else:
        raise EnvironmentError("protocol_failed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1) from None
