from __future__ import annotations

import json
from io import StringIO

from app.agents.contracts import AgentLoopObservationV1
from app.llm.ports import ModelUsage
from app.obs.agent_loop import StructlogAgentLoopObserver
from app.obs.logging import configure_logging


def test_structlog_agent_observer_writes_only_structured_summary_fields() -> None:
    stream = StringIO()
    configure_logging(log_level="INFO", stream=stream)
    observer = StructlogAgentLoopObserver()
    observation = AgentLoopObservationV1(
        event="agent.model.completed",
        sequence=1,
        iteration=1,
        prompt_version="sha256:" + ("a" * 64),
        duration_ms=12.5,
        model_call_count=1,
        tool_call_count=0,
        tool_result_count=0,
        usage=ModelUsage(input_tokens=4, output_tokens=2),
        output_bytes=81,
        proposed_tool_call_count=1,
    )

    observer.observe(observation)

    event = json.loads(stream.getvalue())
    assert event["event"] == "agent.model.completed"
    assert event["schema_version"] == 1
    assert event["sequence"] == 1
    assert event["iteration"] == 1
    assert event["duration_ms"] == 12.5
    assert event["usage"] == {"input_tokens": 4, "output_tokens": 2}
    assert event["output_bytes"] == 81
    assert event["proposed_tool_call_count"] == 1
    assert "content" not in event
    assert "messages" not in event
    assert "arguments" not in event
    assert "result" not in event
    assert "call_id" not in event
    assert "context" not in event


def test_failed_observation_logs_fixed_category_without_exception_text() -> None:
    stream = StringIO()
    configure_logging(log_level="INFO", stream=stream)
    observer = StructlogAgentLoopObserver()
    observer.observe(
        AgentLoopObservationV1(
            event="agent.loop.failed",
            sequence=2,
            iteration=3,
            prompt_version="sha256:" + ("b" * 64),
            duration_ms=3.0,
            model_call_count=2,
            tool_call_count=1,
            tool_result_count=0,
            error_category="tool_error",
        )
    )

    event = json.loads(stream.getvalue())
    assert event["error_category"] == "tool_error"
    assert "error" not in event
    assert "exception" not in event
