"""Explicit opt-in CI partitioning; default pytest runs never load this plugin."""

from collections.abc import Generator, Sequence

import pytest


def select_node_ids(node_ids: Sequence[str], *, index: int, count: int) -> frozenset[str]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("CI shard requires count > 0 and 0 <= index < count")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("CI shard requires unique node IDs")
    return frozenset(sorted(node_ids)[index::count])


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ci-sharding")
    group.addoption("--ci-shard-index", type=int, default=None)
    group.addoption("--ci-shard-count", type=int, default=None)


def pytest_configure(config: pytest.Config) -> None:
    index = config.getoption("ci_shard_index")
    count = config.getoption("ci_shard_count")
    if index is None or count is None:
        raise pytest.UsageError("CI shard requires both --ci-shard-index and --ci-shard-count")
    try:
        select_node_ids((), index=index, count=count)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> Generator[None, object, object]:
    # Resume after native --deselect/-k/-m and other collection hooks have finished.
    result = yield
    if session.testsfailed:
        return result  # Preserve collection failures, never partition a partial suite.
    try:
        selected = select_node_ids(
            [item.nodeid for item in items],
            index=config.getoption("ci_shard_index"),
            count=config.getoption("ci_shard_count"),
        )
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    if not selected:
        raise pytest.UsageError("CI shard selected no tests")
    deselected = [item for item in items if item.nodeid not in selected]
    # Membership is assigned by sorted ID; execution retains the original order.
    items[:] = [item for item in items if item.nodeid in selected]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    return result
