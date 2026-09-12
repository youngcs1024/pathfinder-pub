"""Opt-in teaching session: MCP supplies transport, the local Registry governs calls."""

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path
from time import monotonic

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.domain.tool_invocations import ToolInvocationRecorderPort
from app.domain.tracing import current_trace_scope, finish_trace_span, start_trace_span
from app.tools.contracts import GraphToolPolicy, ToolExecutionContext, ToolInputModel
from app.tools.mcp_experiment._logging import _sdk_logging_guard
from app.tools.mcp_experiment.server import PROTOCOL_TARGET, SERVER_NAME, SERVER_VERSION
from app.tools.registry import ToolRegistry
from app.tools.teaching import (
    LOOKUP_SYNTHETIC_RECORD_SPEC,
    LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
    LookupSyntheticRecordInput,
    LookupSyntheticRecordOutput,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
STARTUP_SECONDS = 5.0
DISCOVERY_SECONDS = 5.0
SHUTDOWN_SECONDS = 8.0
MCP_EXPERIMENT_POLICY = GraphToolPolicy(
    name="mcp_teaching_experiment",
    allowed_tool_names=frozenset({LOOKUP_SYNTHETIC_RECORD_TOOL_NAME}),
    allowed_effects=frozenset({LOOKUP_SYNTHETIC_RECORD_SPEC.effect}),
)


class MCPExperimentError(Exception):
    def __init__(self) -> None:
        super().__init__("MCP experiment lifecycle failed")


class MCPDiscoveryError(MCPExperimentError):
    def __init__(self) -> None:
        Exception.__init__(self, "MCP experiment discovery failed")


class MCPStartupError(MCPExperimentError):
    def __init__(self) -> None:
        Exception.__init__(self, "MCP experiment startup failed")


class MCPCleanupError(MCPExperimentError):
    def __init__(self) -> None:
        Exception.__init__(self, "MCP experiment cleanup failed")


class MCPCallError(MCPExperimentError):
    def __init__(self) -> None:
        Exception.__init__(self, "MCP experiment call failed")


def _launch_parameters() -> StdioServerParameters:
    return StdioServerParameters(
        # Do not resolve the venv symlink: that would select the base interpreter.
        command=str(Path(sys.executable).absolute()),
        args=["-m", "app.tools.mcp_experiment.server"],
        cwd=PROJECT_ROOT,
        # SDK merges its HOME/LOGNAME/PATH/SHELL/TERM/USER subset. NOT an empty env.
        env={},
    )


async def _validated_registry(
    client: Client, recorder: ToolInvocationRecorderPort | None
) -> ToolRegistry:
    """Discover a fixed surface; never turn discovered capabilities into local policy."""
    try:
        info = client.server_info
        if (
            client.protocol_version != PROTOCOL_TARGET
            or info is None
            or info.name != SERVER_NAME
            or info.version != SERVER_VERSION
        ):
            raise MCPDiscoveryError
        listing = await client.list_tools()
        if (
            listing.next_cursor is not None
            or len(listing.tools) != 1
            or listing.tools[0].name != LOOKUP_SYNTHETIC_RECORD_TOOL_NAME
        ):
            raise MCPDiscoveryError
        tool = listing.tools[0]
        schema = tool.input_schema
        properties = schema.get("properties", {})
        if (
            schema.get("type") != "object"
            or set(properties) != {"record_id"}
            or schema.get("required") != ["record_id"]
            or properties["record_id"].get("type") != "string"
            or properties["record_id"].get("maxLength") != 64
            or tool.output_schema != LookupSyntheticRecordOutput.model_json_schema()
        ):
            raise MCPDiscoveryError
        # The teaching server has no other discoverable surface (including pages).
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        prompts = await client.list_prompts()
        if (
            resources.resources
            or resources.next_cursor is not None
            or templates.resource_templates
            or templates.next_cursor is not None
            or prompts.prompts
            or prompts.next_cursor is not None
        ):
            raise MCPDiscoveryError
    except Exception:
        raise MCPDiscoveryError from None

    async def invoke(tool_input: ToolInputModel, _context: ToolExecutionContext) -> object:
        if not isinstance(tool_input, LookupSyntheticRecordInput):
            raise MCPCallError
        scope = current_trace_scope()
        started_at = monotonic()
        span = (
            start_trace_span(scope, span_kind="mcp_transport", metadata={"transport": "stdio"})
            if scope is not None
            else None
        )
        status, category = "succeeded", None
        try:
            result = await client.call_tool(
                LOOKUP_SYNTHETIC_RECORD_TOOL_NAME, {"record_id": tool_input.record_id}
            )
            if result.is_error:
                raise MCPCallError
            # Untrusted result. Registry owns final strict validation and the byte cap.
            return result.structured_content
        except asyncio.CancelledError:
            status, category = "cancelled", "cancelled"
            raise
        except Exception:
            status, category = "failed", "mcp_call_failed"
            raise MCPCallError from None
        finally:
            if scope is not None:
                finish_trace_span(
                    scope, span, started_at=started_at, status=status, error_category=category
                )

    return ToolRegistry(
        specs=(replace(LOOKUP_SYNTHETIC_RECORD_SPEC, handler=invoke),),
        policies=(MCP_EXPERIMENT_POLICY,),
        recorder=recorder,
    )


@asynccontextmanager
async def open_mcp_experiment(
    *, recorder: ToolInvocationRecorderPort | None = None
) -> AsyncIterator[ToolRegistry]:
    """Open once, validate once, reuse until exit; no caller-controlled launch settings.

    The SDK owns all process/pipe handling. Budgets are separate from Registry calls.
    SDK shutdown retains process ownership; cleanup failure takes precedence over body errors.
    """
    stack = AsyncExitStack()
    # Discard child stderr without an accumulating buffer or a blocking stderr pipe.
    with _sdk_logging_guard() as evidence, open(os.devnull, "w", encoding="utf-8") as errlog:
        try:
            try:
                async with asyncio.timeout(STARTUP_SECONDS):
                    streams = await stack.enter_async_context(
                        stdio_client(_launch_parameters(), errlog=errlog)
                    )
            except Exception:
                raise MCPStartupError from None

            # Hand the already-open official streams to Client's public Transport API.
            # This does not parse/wrap bytes or take ownership of the child process.
            @asynccontextmanager
            async def connected_transport():
                yield streams

            try:
                async with asyncio.timeout(DISCOVERY_SECONDS):
                    client = await stack.enter_async_context(
                        Client(
                            connected_transport(),
                            mode="auto",
                            cache=None,
                            input_required_max_rounds=0,
                        )
                    )
                    registry = await _validated_registry(client, recorder)
            except Exception:
                raise MCPDiscoveryError from None
            yield registry
        finally:
            try:
                async with asyncio.timeout(SHUTDOWN_SECONDS):
                    await stack.aclose()
                if evidence.survivor:
                    raise MCPCleanupError
            except Exception:
                raise MCPCleanupError from None
