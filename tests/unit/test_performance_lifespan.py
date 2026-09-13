"""Real Uvicorn signal regression without Docker or application database setup."""

import select
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from tests.performance.metrics import Collector, Packet
from tests.performance.metrics_runtime import instrument_application

CHILD = """
import asyncio, sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import uvicorn
from fastapi import FastAPI
from tests.performance.metrics import Collector
from tests.performance.metrics_runtime import instrument_application

@asynccontextmanager
async def lifespan(app):
    print("ready", flush=True)
    yield

app = FastAPI(lifespan=lifespan)
collector = Collector("api", Path(sys.argv[1]))
instrument_application(app, SimpleNamespace(create_database_engine=lambda: None), collector)
asyncio.run(uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None)).serve())
"""


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_real_uvicorn_signal_preserves_flush_boundary(tmp_path, sig):
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert select.select([child.stdout], [], [], 15)[0]
        assert child.stdout.readline() == b"ready\n"
        child.send_signal(sig)
        assert child.wait(timeout=15) == -sig
        path = tmp_path / "metrics-api.json"
        if sig == signal.SIGTERM:
            packet = Packet.model_validate_json(path.read_bytes())
            assert packet.role == "api" and not packet.write_failed
        else:
            assert not path.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        child.stdout.close()


@pytest.mark.asyncio
async def test_startup_failure_finalizes_and_restores(tmp_path):
    @asynccontextmanager
    async def lifespan(app):
        raise RuntimeError("PRIVATE_CANARY")
        yield

    root = SimpleNamespace(create_database_engine=lambda: None)
    original = root.create_database_engine
    app = FastAPI(lifespan=lifespan)
    collector = Collector("api", tmp_path)
    instrument_application(app, root, collector)
    with pytest.raises(RuntimeError):
        async with app.router.lifespan_context(app):
            pass
    assert root.create_database_engine is original
    before = (tmp_path / "metrics-api.json").read_bytes()
    collector.finish()
    assert not collector.write_failed
    assert (tmp_path / "metrics-api.json").read_bytes() == before
    assert b"PRIVATE_CANARY" not in before


def test_finalization_write_failure_is_retained(tmp_path):
    collector = Collector("api", tmp_path)
    (tmp_path / "metrics-api.json").write_text("existing")
    collector.finish()
    assert collector.write_failed
    collector.finish()
    assert collector.write_failed
    assert (tmp_path / "metrics-api.json").read_text() == "existing"
