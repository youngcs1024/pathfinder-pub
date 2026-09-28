"""Small D CI matrix runs the same worker, subprocess and connection fault harness."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from app.db.session import create_database_engine, create_session_factory
from tests.evals.product_acceptance_contracts import publish
from tests.evals.resume_recovery_contracts import cases
from tests.evals.resume_recovery_faults import ledger, run_case
from tests.evals.resume_recovery_fixture import prepare, snapshot

pytestmark = pytest.mark.integration
SMALL = [
    c
    for c in cases()
    if (c["kind"] == "controlled" and c["ordinal"] == 0)
    or (c["kind"] == "real" and c["ordinal"] == 90)
    or (c["kind"] == "blocking" and c["ordinal"] in (0, 1))
]


@pytest.mark.parametrize("case", SMALL, ids=[c["case_id"] for c in SMALL])
async def test_d_worker_fault_matrix(migrated_database_url, tmp_path: Path, case):
    root = tmp_path / case["case_id"]
    root.mkdir(mode=0o700)
    engine = create_database_engine(SecretStr(migrated_database_url))
    try:
        sessions = create_session_factory(engine)
        state = await prepare(sessions, root, case)
        result = await run_case(migrated_database_url, sessions, root, state)
        assert result["status"] == "PASS", result
        assert result["old_owner_rejected"] and result["duplicate_side_effects"] == 0
        assert result["model_attempts"] == len(await ledger(sessions, state))
        publish(root / "result.json", result)
        assert await prepare(sessions, root, case) == state
        before = await snapshot(sessions, state)
        # Controller restart with a published result must not need another invocation.
        replayed = await run_case(migrated_database_url, sessions, root, state)
        assert replayed["status"] == "PASS"
        assert await snapshot(sessions, state) == before
        assert replayed["model_attempts"] == result["model_attempts"]
    finally:
        await engine.dispose()
