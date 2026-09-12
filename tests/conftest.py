"""Shared offline observation fixtures."""

import pytest

from tests.tracing import CollectingTraceSink


@pytest.fixture
def collecting_trace_sink() -> CollectingTraceSink:
    return CollectingTraceSink()
