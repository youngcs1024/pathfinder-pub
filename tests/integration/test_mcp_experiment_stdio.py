"""Real stdio normal path; intentionally does not request PostgreSQL fixtures."""

import json
from time import monotonic
from types import SimpleNamespace
from uuid import uuid4

from app.llm.ports import ModelToolCall
from app.tools.contracts import ToolRunContext
from app.tools.mcp_experiment.client import MCP_EXPERIMENT_POLICY, open_mcp_experiment
from app.tools.teaching import (
    LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
    LookupSyntheticRecordInput,
    resolve_synthetic_record,
)


async def test_real_stdio_registry_calls_close_and_reopen():
    for cases in (("record-1", "missing-record", "record-2"), ("x" * 64,)):
        async with open_mcp_experiment() as registry:
            runtime = registry.bind(
                policy_name=MCP_EXPERIMENT_POLICY.name,
                context=ToolRunContext(
                    workspace_id=uuid4(),
                    actor_user_id=uuid4(),
                    run_id=uuid4(),
                    action_intent_id=None,
                    approval_request_id=None,
                    trusted_target={"target": "private-context-canary"},
                    deadline=monotonic() + 10.0,
                    cancellation=SimpleNamespace(is_cancelled=lambda: False),
                ),
            )
            for record_id in cases:
                result = await runtime.execute(
                    ModelToolCall(
                        call_id=str(uuid4()),
                        name=LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
                        arguments={"record_id": record_id},
                    )
                )
                assert (
                    json.loads(result)
                    == resolve_synthetic_record(
                        LookupSyntheticRecordInput(record_id=record_id)
                    ).model_dump()
                )
