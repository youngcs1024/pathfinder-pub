from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_CONTENT_SECURITY_POLICY_HEADER = "Content-Security-Policy"
_PRODUCT_UI_PATH = "/"


def product_ui_content_security_policy(*, browser_connect_origin: str | None) -> str:
    connect_sources = "'self'"
    if browser_connect_origin is not None:
        connect_sources = f"{connect_sources} {browser_connect_origin}"
    return "; ".join(
        (
            "default-src 'none'",
            "base-uri 'none'",
            "object-src 'none'",
            "frame-ancestors 'none'",
            "form-action 'self'",
            "script-src 'self'",
            "style-src 'self'",
            f"connect-src {connect_sources}",
        )
    )


class BrowserSecurityHeadersMiddleware:
    """Add stateless browser hardening without inspecting product state."""

    def __init__(self, app: ASGIApp, *, browser_connect_origin: str | None) -> None:
        self._app = app
        self._product_ui_csp = product_ui_content_security_policy(
            browser_connect_origin=browser_connect_origin
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Frame-Options"] = "DENY"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "no-referrer"
                if scope["path"] == _PRODUCT_UI_PATH:
                    headers[_CONTENT_SECURITY_POLICY_HEADER] = self._product_ui_csp
            await send(message)

        await self._app(scope, receive, send_with_security_headers)
