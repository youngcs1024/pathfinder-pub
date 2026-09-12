from __future__ import annotations

from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI

from app.config import Settings
from app.main import create_app

_PUBLISHABLE_KEY = "sb_publishable_test-public-key"
_BASIC_SECURITY_HEADERS = {
    "x-frame-options": "DENY",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
}


class _CallSentinel:
    def __init__(self) -> None:
        self.calls = 0

    def __getattr__(self, _name: str) -> object:
        self.calls += 1
        raise AssertionError("product dependency reached during CORS preflight")


async def _request(
    application: FastAPI,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, headers=headers)


def _assert_basic_security_headers(response: httpx.Response) -> None:
    for name, value in _BASIC_SECURITY_HEADERS.items():
        assert response.headers[name] == value


async def test_fake_product_ui_has_strict_csp_and_basic_security_headers() -> None:
    response = await _request(create_app(Settings(log_level="ERROR")), "GET", "/")

    assert response.status_code == 200
    _assert_basic_security_headers(response)
    policy = response.headers["content-security-policy"]
    for directive in (
        "default-src 'none'",
        "base-uri 'none'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "connect-src 'self'",
    ):
        assert directive in policy
    for forbidden in ("*", "'unsafe-inline'", "'unsafe-eval'"):
        assert forbidden not in policy


async def test_supabase_csp_uses_only_exact_validated_browser_origin() -> None:
    database_canary = "DB-CREDENTIAL-CANARY"
    provider_canary = "QWEN-SECRET-CANARY"
    arbitrary_origin = "https://browser.example"
    application = create_app(
        Settings(
            auth_mode="supabase",
            supabase_project_ref="abcdefghijklmnopqrst",
            supabase_publishable_key=_PUBLISHABLE_KEY,
            database_url=(
                f"postgresql+psycopg://pathfinder:{database_canary}@127.0.0.1:5432/pathfinder"
            ),
            qwen_api_key=provider_canary,
            log_level="ERROR",
        )
    )

    response = await _request(
        application,
        "GET",
        "/",
        headers={"Origin": arbitrary_origin, "Authorization": "Bearer access-token-canary"},
    )

    policy = response.headers["content-security-policy"]
    assert "connect-src 'self' https://abcdefghijklmnopqrst.supabase.co" in policy
    for forbidden in (
        database_canary,
        provider_canary,
        _PUBLISHABLE_KEY,
        "jwks",
        "service_role",
        "sb_secret",
        "access-token-canary",
        "refresh_token",
        arbitrary_origin,
    ):
        assert forbidden not in policy
    assert "access-control-allow-origin" not in response.headers


async def test_static_assets_keep_mime_types_with_nosniff() -> None:
    application = create_app(Settings(log_level="ERROR"))

    javascript = await _request(application, "GET", "/static/pathfinder.js")
    stylesheet = await _request(application, "GET", "/static/pathfinder.css")

    assert javascript.status_code == 200
    assert javascript.headers["content-type"].startswith(
        ("text/javascript", "application/javascript")
    )
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    _assert_basic_security_headers(javascript)
    _assert_basic_security_headers(stylesheet)


async def test_problem_details_and_request_id_survive_security_middleware() -> None:
    application = create_app(Settings(log_level="ERROR"))

    @application.get("/browser-security/items/{item_id}")
    async def get_item(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    missing = await _request(application, "GET", "/missing")
    invalid = await _request(application, "GET", "/browser-security/items/not-an-integer")

    assert missing.status_code == 404
    assert missing.headers["content-type"] == "application/problem+json"
    assert missing.json() == {
        "type": "about:blank",
        "title": "Not Found",
        "status": 404,
        "detail": "The requested resource was not found.",
        "request_id": missing.headers["x-request-id"],
    }
    assert invalid.status_code == 422
    assert invalid.headers["content-type"] == "application/problem+json"
    assert invalid.json() == {
        "type": "urn:pathfinder:problem:request-validation",
        "title": "Unprocessable Content",
        "status": 422,
        "detail": "The request did not satisfy the required schema.",
        "request_id": invalid.headers["x-request-id"],
    }
    for response in (missing, invalid):
        UUID(response.headers["x-request-id"])
        _assert_basic_security_headers(response)


async def test_hostile_read_and_mutation_preflights_stop_before_product_dependencies() -> None:
    application = create_app(Settings(log_level="ERROR"))
    actor = _CallSentinel()
    tenant = _CallSentinel()
    runs = _CallSentinel()
    application.state.actor_provider = actor
    application.state.tenant_service = tenant
    application.state.run_service = runs
    workspace_id = uuid4()

    cases = (
        (
            "/api/v1/me",
            "GET",
            "authorization",
        ),
        (
            f"/api/v1/workspaces/{workspace_id}/runs",
            "POST",
            "authorization,content-type",
        ),
    )
    for path, requested_method, requested_headers in cases:
        response = await _request(
            application,
            "OPTIONS",
            path,
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": requested_method,
                "Access-Control-Request-Headers": requested_headers,
            },
        )
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers
        assert "access-control-allow-credentials" not in response.headers
        assert "https://evil.example" not in " ".join(response.headers.values())
        _assert_basic_security_headers(response)

    assert actor.calls == tenant.calls == runs.calls == 0


async def test_hostile_simple_request_gets_no_cross_origin_disclosure_grant() -> None:
    response = await _request(
        create_app(Settings(log_level="ERROR")),
        "GET",
        "/",
        headers={"Origin": "https://evil.example"},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-credentials" not in response.headers


async def test_developer_docs_and_openapi_remain_available_without_weakening_product_csp() -> None:
    application = create_app(Settings(log_level="ERROR"))

    docs = await _request(application, "GET", "/docs")
    openapi = await _request(application, "GET", "/openapi.json")
    product_ui = await _request(application, "GET", "/")

    assert docs.status_code == 200
    assert openapi.status_code == 200
    assert "content-security-policy" not in docs.headers
    assert "content-security-policy" not in openapi.headers
    assert "'unsafe-inline'" not in product_ui.headers["content-security-policy"]
    assert "'unsafe-eval'" not in product_ui.headers["content-security-policy"]
    _assert_basic_security_headers(docs)
    _assert_basic_security_headers(openapi)
