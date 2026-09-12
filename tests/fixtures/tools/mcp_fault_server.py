"""Fixed test-only subprocess faults; never imported by the teaching entrypoint."""

import json
import os
import signal
import sys
from pathlib import Path

# Record process identity before SDK imports. /proc start time protects against PID reuse.
MODE, DIRECTORY = sys.argv[1:]
ROOT = Path(DIRECTORY)
ROOT.joinpath("pid").write_text(
    json.dumps({"pid": os.getpid(), "start": Path("/proc/self/stat").read_text().split()[21]})
)
CANARY = "GATE12_RAW_SECRET_CANARY"


def pause() -> None:
    while True:
        signal.pause()


def main() -> None:
    if MODE == "early_exit":
        return
    if MODE == "silent_discovery":
        pause()
    if MODE in {"malformed_json", "stdout_flood", "stdout_unterminated"}:
        payload = (
            (CANARY + "\n").encode()
            if MODE == "malformed_json"
            else (CANARY * 32768 + ("\n" if MODE == "stdout_flood" else "")).encode()
        )
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
        ROOT.joinpath("emitted").touch()
        pause()

    import anyio
    from mcp.types import CallToolResult, TextContent

    from app.tools.mcp_experiment.server import build_server

    async def fault(ctx, call_next):
        if ctx.method == "tools/call":
            ROOT.joinpath("called").touch()
            if MODE == "call_hang":
                await anyio.sleep_forever()
            if MODE == "crash_on_call":
                os._exit(23)
            if MODE == "malformed_call":
                sys.stdout.write(CANARY + "\n")
                sys.stdout.flush()
                await anyio.sleep_forever()
            if MODE == "stderr_flood":
                sys.stderr.write(CANARY * 32768)
                sys.stderr.flush()
            if MODE == "env_probe":
                # Names only, including interpreter-synthesized keys if any.
                inherited = [
                    entry.split(b"=", 1)[0].decode()
                    for entry in Path("/proc/self/environ").read_bytes().split(b"\0")
                    if entry
                ]
                ROOT.joinpath("env").write_text(
                    json.dumps(
                        {
                            "inherited_keys": sorted(inherited),
                            "runtime_keys": sorted(os.environ),
                            "python_locale_coercion": os.environ.get("LC_CTYPE") == "C.UTF-8",
                        }
                    )
                )
            if MODE == "oversized_output":
                output = {"record_id": "record-1", "found": True, "summary": "x" * 4096}
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output))],
                    structured_content=output,
                )
        return await call_next(ctx)

    server = build_server()
    server.middleware.append(fault)
    if MODE == "linger_on_shutdown":

        def term(_signum, _frame):
            ROOT.joinpath("term").touch()

        signal.signal(signal.SIGTERM, term)
    server.run(transport="stdio")
    if MODE == "linger_on_shutdown":
        ROOT.joinpath("stdin_closed").touch()
        pause()


if __name__ == "__main__":
    main()
