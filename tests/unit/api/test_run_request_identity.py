from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.datastructures import Headers

from app.api.errors import InvalidIdempotencyKeyError, install_exception_handlers
from app.api.run_request_identity import parse_idempotency_key

KEY = "12345678-1234-4234-9234-123456789abc"


def test_absent_key_is_legacy_and_valid_key_is_canonical() -> None:
    assert parse_idempotency_key([]) is None
    for value in (KEY, KEY.upper()):
        parsed = parse_idempotency_key([value])
        assert parsed == UUID(KEY)
        assert str(parsed) == KEY
        assert parsed.version == 4


@pytest.mark.parametrize(
    "values",
    [
        [""],
        [" "],
        [KEY, KEY],
        [KEY, KEY.upper()],
        [f"{KEY},{KEY}"],
        [f"{KEY}, {KEY}"],
        [f" {KEY}"],
        [f"{KEY}\n"],
        [KEY.replace("-", "")],
        [f"urn:uuid:{KEY}"],
        [f"{{{KEY}}}"],
        ["12345678-1234-1234-9234-123456789abc"],
        ["12345678-1234-4234-7234-123456789abc"],
        ["00000000-0000-0000-0000-000000000000"],
        ["CANARY"],
        [None],
        KEY,
    ],
)
def test_invalid_header_has_fixed_400(values: object) -> None:
    with pytest.raises(InvalidIdempotencyKeyError) as caught:
        parse_idempotency_key(values)
    assert caught.value.status_code == 400
    assert caught.value.detail == "Idempotency-Key must be a single UUID4."
    assert "CANARY" not in str(caught.value)


def test_raw_duplicate_headers_are_not_collapsed() -> None:
    headers = Headers(raw=[(b"idempotency-key", KEY.encode())] * 2)
    with pytest.raises(InvalidIdempotencyKeyError):
        parse_idempotency_key(headers.getlist("Idempotency-Key"))


async def test_parser_error_uses_safe_problem_handler_in_test_only_route() -> None:
    application = FastAPI()
    install_exception_handlers(application)

    @application.post("/parser-contract")
    async def parser_contract(request: Request) -> dict[str, str]:
        parse_idempotency_key(request.headers.getlist("Idempotency-Key"))
        return {"result": "parsed"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://testserver"
    ) as client:
        response = await client.post("/parser-contract", headers={"Idempotency-Key": "CANARY"})
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "urn:pathfinder:problem:invalid-idempotency-key"
    assert response.json()["detail"] == "Idempotency-Key must be a single UUID4."
    assert "CANARY" not in response.text
