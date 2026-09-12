"""One experiment-only stdio tool. Local Registry remains the authority."""

from typing import Annotated

from mcp.server import MCPServer
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field, ValidationError

from app.tools.teaching import (
    LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
    LookupSyntheticRecordInput,
    LookupSyntheticRecordOutput,
    resolve_synthetic_record,
)

SERVER_NAME = "pathfinder-gate12-teaching"
SERVER_VERSION = "0.1.0"
PROTOCOL_TARGET = "2026-07-28"
INPUT_FAILURE = "lookup-input failure"
LOOKUP_FAILURE = "synthetic-lookup failure"


async def _validate_lookup_arguments(
    ctx: ServerRequestContext, call_next: CallNext
) -> HandlerResult:
    # SDK 2.1.1 ignores extra function arguments and includes Pydantic input values
    # in validation errors. Validate before that path, using its public middleware
    # hook. This is post-parse validation, NOT a raw transport cap or authorization.
    if ctx.method == "tools/call":
        params = ctx.params
        if not isinstance(params, dict) or params.get("name") != LOOKUP_SYNTHETIC_RECORD_TOOL_NAME:
            return CallToolResult(
                is_error=True, content=[TextContent(type="text", text=INPUT_FAILURE)]
            )
        try:
            LookupSyntheticRecordInput.model_validate(params.get("arguments"), strict=True)
        except ValidationError:
            return CallToolResult(
                is_error=True, content=[TextContent(type="text", text=INPUT_FAILURE)]
            )
    return await call_next(ctx)


def build_server() -> MCPServer:
    server = MCPServer(SERVER_NAME, version=SERVER_VERSION, middleware=[_validate_lookup_arguments])

    @server.tool(
        name=LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    async def lookup_record(
        record_id: Annotated[str, Field(strict=True, max_length=64)],
    ) -> LookupSyntheticRecordOutput:
        """Look up one deterministic synthetic learning record."""
        try:
            tool_input = LookupSyntheticRecordInput(record_id=record_id)
        except ValidationError:
            raise ToolError(INPUT_FAILURE) from None
        try:
            return LookupSyntheticRecordOutput.model_validate(
                resolve_synthetic_record(tool_input), strict=True
            )
        except Exception:
            # A deliberate SDK ToolError avoids its unexpected-error traceback log.
            # BaseException (including cancellation) must propagate unchanged.
            raise ToolError(LOOKUP_FAILURE) from None

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
