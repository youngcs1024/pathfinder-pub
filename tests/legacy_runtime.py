"""Explicit historical execution bindings for frozen Gate/E regressions only."""

from sqlalchemy import true

from app.db.jobs import SqlAlchemyWorkerJobStore as _Jobs
from app.db.run_execution import SqlAlchemyRunExecutionReader as _Reader
from app.db.runs import SqlAlchemyRunStore as _Runs
from app.domain.run_payloads import LEGACY_GRAPH_VERSION, LEGACY_READ_CONTRACTS

LEGACY_EXECUTION_CONTRACTS = tuple(
    c for c in LEGACY_READ_CONTRACTS if c.graph_version == LEGACY_GRAPH_VERSION
)


class SqlAlchemyWorkerJobStore(_Jobs):
    def _executable_run_predicate(self):
        # Historical claimer semantics, including fail-closed unknown-version tests.
        return true()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, execution_contracts=LEGACY_EXECUTION_CONTRACTS)


class SqlAlchemyRunExecutionReader(_Reader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, execution_contracts=LEGACY_EXECUTION_CONTRACTS)


class SqlAlchemyRunStore(_Runs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, execution_contracts=LEGACY_EXECUTION_CONTRACTS)
