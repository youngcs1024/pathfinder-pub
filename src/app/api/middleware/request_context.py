from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import REQUEST_ID_HEADER
from app.obs.logging import (
    bind_request_context,
    clear_logging_context,
    get_logger,
)


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    template = getattr(route, "path", None)
    return template if isinstance(template, str) else "<unmatched>"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._logger = get_logger("app.http")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        clear_logging_context()
        request_id = str(uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        bind_request_context(request_id=request_id)
        started_at = perf_counter()
        status_code: int | None = None

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self._app(scope, receive, send_with_request_id)
        except Exception as error:
            self._logger.error(
                "http.request.failed",
                method=scope["method"],
                route=_route_template(scope),
                status_code=500,
                duration_ms=round((perf_counter() - started_at) * 1000, 3),
                error_type=type(error).__name__,
            )
            raise
        else:
            self._logger.info(
                "http.request.completed",
                method=scope["method"],
                route=_route_template(scope),
                status_code=status_code or 500,
                duration_ms=round((perf_counter() - started_at) * 1000, 3),
            )
        finally:
            clear_logging_context()
