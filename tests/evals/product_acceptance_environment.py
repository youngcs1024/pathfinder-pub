"""Owned loopback database for E8.3; never accepts a caller-supplied DSN.

Containers are stopped and retained, not removed. Evidence and data are not deleted.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from uuid import uuid4

import psycopg
from alembic import command as alembic
from alembic.config import Config

from tests.evals.product_acceptance_contracts import AcceptanceError, publish, require
from tests.evals.quality_experiment_binding import ROOT

IMAGE = "pgvector/pgvector:0.8.5-pg16"
LABEL = "pathfinder.e83.owner"


def docker(*args, env=None):
    try:
        result = subprocess.run(
            ["docker", *args],
            env=env,
            capture_output=True,
            timeout=30,
            check=False,
        )
        require(result.returncode == 0 and len(result.stdout) < 1_000_000, "docker_failed")
        return result.stdout.decode().strip()
    except AcceptanceError:
        raise
    except Exception:
        raise AcceptanceError("docker_failed") from None


def local_docker():
    require(os.environ.get("DOCKER_CONTEXT") in {None, "default"}, "remote_docker_rejected")
    host = os.environ.get("DOCKER_HOST")
    require(host is None or host.startswith("unix:///"), "remote_docker_rejected")
    context = json.loads(docker("context", "inspect"))[0]
    endpoint = host or context["Endpoints"]["docker"]["Host"]
    require(endpoint.startswith("unix:///"), "remote_docker_rejected")


def validate_owned(info, *, owner, container_id, running):
    require(
        info["Id"] == container_id and info["Config"]["Labels"].get(LABEL) == owner,
        "ownership_mismatch",
    )
    require(info["Name"] == f"/pf-e83-{owner}", "ownership_mismatch")
    host = info["HostConfig"]
    require(
        not host.get("Privileged")
        and not host.get("Binds")
        and host["NetworkMode"] == "bridge"
        and host["NanoCpus"] == 2_000_000_000
        and host["Memory"] == 2_147_483_648
        and host["PidsLimit"] == 128,
        "unsafe_container",
    )
    if running:
        require(info["State"]["Running"], "database_not_running")
        ports = info["NetworkSettings"]["Ports"].get("5432/tcp", [])
        require(len(ports) == 1 and ports[0]["HostIp"] == "127.0.0.1", "unsafe_binding")
        port = int(ports[0]["HostPort"])
        require(0 < port < 65536, "unsafe_binding")
        return port
    return None


class OwnedProductDatabase:
    def __init__(self, root, image_id):
        self.root, self.image_id = root, image_id
        self.owner, self.container_id = uuid4().hex, None
        self.cleanup_ok = False

    def inspect(self, *, running):
        info = json.loads(docker("inspect", self.container_id))[0]
        require(info["Image"] == self.image_id, "image_changed")
        return validate_owned(
            info, owner=self.owner, container_id=self.container_id, running=running
        )

    async def __aenter__(self):
        local_docker()
        require(
            docker("image", "inspect", IMAGE, "--format", "{{.Id}}") == self.image_id,
            "image_changed",
        )
        password = uuid4().hex
        database = f"pathfinder_test_{self.owner}"
        environment = {**os.environ, "POSTGRES_PASSWORD": password}
        try:
            self.container_id = docker(
                "create",
                "--name",
                f"pf-e83-{self.owner}",
                "--label",
                f"{LABEL}={self.owner}",
                "--network",
                "bridge",
                "--cpus",
                "2",
                "--memory",
                "2g",
                "--pids-limit",
                "128",
                "--publish",
                "127.0.0.1::5432",
                "--env",
                "POSTGRES_PASSWORD",
                "--env",
                "POSTGRES_USER=pf_e83",
                "--env",
                f"POSTGRES_DB={database}",
                self.image_id,
                env=environment,
            )
            publish(
                self.root / "database-owner.json",
                {
                    "owner": self.owner,
                    "container_id": self.container_id,
                    "image_id": self.image_id,
                    "cpu": 2,
                    "memory_bytes": 2147483648,
                    "binding": "loopback",
                },
            )
            docker("start", self.container_id)
            port = self.inspect(running=True)
            for _ in range(60):
                try:
                    with psycopg.connect(
                        host="127.0.0.1",
                        port=port,
                        user="pf_e83",
                        password=password,
                        dbname=database,
                        connect_timeout=2,
                    ):
                        break
                except psycopg.OperationalError:
                    await asyncio.sleep(1)
            else:
                raise AcceptanceError("database_startup_timeout")
            url = f"postgresql+psycopg://pf_e83:{password}@127.0.0.1:{port}/{database}"
            config = Config(str(ROOT / "alembic.ini"))
            config.attributes["database_url"] = url
            alembic.upgrade(config, "head")
            return url
        except BaseException:
            self.stop()
            raise

    def stop(self):
        try:
            if self.container_id is not None:
                self.inspect(running=False)
                docker("stop", "--time", "15", self.container_id)
                info = json.loads(docker("inspect", self.container_id))[0]
                require(not info["State"]["Running"], "cleanup_failed")
            self.cleanup_ok = True
        finally:
            publish(
                self.root / "cleanup.json",
                {
                    "stopped": self.cleanup_ok,
                    "container_retained": self.container_id is not None,
                    "files_deleted": False,
                },
            )

    async def __aexit__(self, *args):
        self.stop()
