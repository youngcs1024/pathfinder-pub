"""Resume the same owned database; a missing receipt never creates replacement state."""

import json
from urllib.parse import quote

import psycopg

from tests.evals.product_acceptance_contracts import publish, read_private_json, require
from tests.evals.product_acceptance_environment import OwnedProductDatabase, docker, local_docker


class ResumableDatabase(OwnedProductDatabase):
    async def __aenter__(self):
        local_docker()
        owner_file = self.root / "database-owner.json"
        if not owner_file.exists():
            require(not (self.root / "database-started.json").exists(), "database_receipt_missing")
            publish(self.root / "database-started.json", {"image_id": self.image_id})
            return await super().__aenter__()
        saved = read_private_json(owner_file)
        self.owner, self.container_id = saved["owner"], saved["container_id"]
        require(saved["image_id"] == self.image_id, "image_changed")
        self.inspect(running=False)
        docker("start", self.container_id)
        port = self.inspect(running=True)
        info = json.loads(docker("inspect", self.container_id))[0]
        environment = dict(value.split("=", 1) for value in info["Config"]["Env"] if "=" in value)
        # Infrastructure credentials stay in memory and never enter evidence files.
        url = (
            f"postgresql+psycopg://pf_e83:{quote(environment['POSTGRES_PASSWORD'], safe='')}"
            f"@127.0.0.1:{port}/{environment['POSTGRES_DB']}"
        )
        import asyncio

        for _ in range(30):
            try:
                with psycopg.connect(
                    url.replace("postgresql+psycopg:", "postgresql:"), connect_timeout=2
                ):
                    return url
            except psycopg.OperationalError:
                await asyncio.sleep(1)
        self.stop()
        raise ValueError("database_startup_timeout")

    def stop(self):
        if self.container_id is not None:
            self.inspect(running=False)
            docker("stop", "--time", "15", self.container_id)
            info = json.loads(docker("inspect", self.container_id))[0]
            require(not info["State"]["Running"], "cleanup_failed")
        self.cleanup_ok = True
