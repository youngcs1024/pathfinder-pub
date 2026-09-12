from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import jwt
import jwt.jwk_set_cache
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientConnectionError

from app.auth.contracts import ActorContext, VerifiedAuthPrincipal
from app.auth.errors import (
    AccessTokenVerificationError,
    JwksUnavailableError,
    TokenVerificationReason,
)
from app.auth.supabase import (
    SUPABASE_JWKS_HTTP_TIMEOUT_SECONDS,
    SUPABASE_JWKS_TTL_SECONDS,
    SUPABASE_JWT_ALGORITHMS,
    SUPABASE_JWT_LEEWAY_SECONDS,
    SupabaseActorProvider,
    SupabaseJwtVerifier,
)
from app.config import Settings
from app.domain.provisioning import (
    ProvisionedPersonalWorkspace,
    ProvisioningService,
    WorkspaceKind,
    WorkspaceRole,
)

PROJECT_REF = "abcdefghijklmnopqrst"
ISSUER = f"https://{PROJECT_REF}.supabase.co/auth/v1"
AUDIENCE = "authenticated"


@dataclass(frozen=True)
class _KeyMaterial:
    kid: str
    private_key: ec.EllipticCurvePrivateKey
    jwk: dict[str, Any]


def _key(kid: str) -> _KeyMaterial:
    private_key = ec.generate_private_key(ec.SECP256R1())
    jwk = jwt.algorithms.ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk.update({"kid": kid, "alg": "ES256", "use": "sig"})
    return _KeyMaterial(kid=kid, private_key=private_key, jwk=jwk)


@pytest.fixture(scope="module")
def key_a() -> _KeyMaterial:
    return _key("key-a")


@pytest.fixture(scope="module")
def key_b() -> _KeyMaterial:
    return _key("key-b")


@pytest.fixture(scope="module")
def key_other() -> _KeyMaterial:
    return _key("key-other")


class _LocalPyJWKClient(PyJWKClient):
    def __init__(self, responses: list[Mapping[str, Any] | BaseException]) -> None:
        super().__init__(
            f"{ISSUER}/.well-known/jwks.json",
            cache_jwk_set=True,
            lifespan=SUPABASE_JWKS_TTL_SECONDS,
            cache_keys=False,
            timeout=SUPABASE_JWKS_HTTP_TIMEOUT_SECONDS,
        )
        self._responses = responses
        self.fetch_count = 0

    def fetch_data(self) -> Any:
        index = self.fetch_count
        self.fetch_count += 1
        response = self._responses[min(index, len(self._responses) - 1)]
        if isinstance(response, BaseException):
            raise response
        data = dict(response)
        assert self.jwk_set_cache is not None
        self.jwk_set_cache.put(data)
        return data


def _settings() -> Settings:
    return Settings(
        auth_mode="supabase",
        supabase_project_ref=PROJECT_REF,
        supabase_publishable_key="sb_publishable_test-public-key",
    )


def _verifier(client: PyJWKClient) -> SupabaseJwtVerifier:
    return SupabaseJwtVerifier(_settings(), jwk_client=client)


def _claims(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    claims: dict[str, object] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": now + timedelta(minutes=5),
        "nbf": now - timedelta(seconds=1),
        "sub": "supabase-user-subject",
    }
    claims.update(overrides)
    return claims


def _token(
    key: _KeyMaterial,
    *,
    claims: Mapping[str, object] | None = None,
    headers: Mapping[str, object] | None = None,
) -> str:
    resolved_headers: dict[str, object] = {"kid": key.kid}
    if headers is not None:
        resolved_headers.update(headers)
    return jwt.encode(
        dict(claims or _claims()),
        key.private_key,
        algorithm="ES256",
        headers=resolved_headers,
    )


def _jwks(*keys: _KeyMaterial) -> dict[str, object]:
    return {"keys": [key.jwk for key in keys]}


def _replace_algorithm_header(token: str, algorithm: str) -> str:
    encoded_header, encoded_payload, encoded_signature = token.split(".")
    padded_header = encoded_header + "=" * (-len(encoded_header) % 4)
    header = json.loads(base64.urlsafe_b64decode(padded_header))
    header["alg"] = algorithm
    replacement = base64.urlsafe_b64encode(
        json.dumps(header, separators=(",", ":")).encode()
    ).rstrip(b"=")
    return b".".join((replacement, encoded_payload.encode(), encoded_signature.encode())).decode()


async def _assert_invalid(
    verifier: SupabaseJwtVerifier,
    token: str,
    reason: TokenVerificationReason,
) -> AccessTokenVerificationError:
    with pytest.raises(AccessTokenVerificationError) as captured:
        await verifier.verify_access_token(token)
    assert captured.value.reason is reason
    return captured.value


async def test_valid_es256_token_returns_only_exact_verified_subject(
    key_a: _KeyMaterial,
) -> None:
    client = _LocalPyJWKClient([_jwks(key_a)])

    principal = await _verifier(client).verify_access_token(_token(key_a))

    assert principal == VerifiedAuthPrincipal(subject="supabase-user-subject")
    assert repr(principal).startswith("<app.auth.contracts.VerifiedAuthPrincipal object at ")
    assert client.fetch_count == 1


class _ActorVerifier:
    def __init__(self, outcome: VerifiedAuthPrincipal | BaseException) -> None:
        self.outcome = outcome
        self.tokens: list[str] = []

    async def verify_access_token(self, token: str) -> VerifiedAuthPrincipal:
        self.tokens.append(token)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _ActorProvisioningStore:
    def __init__(self, outcome: ProvisionedPersonalWorkspace | BaseException) -> None:
        self.outcome = outcome
        self.subjects: list[str] = []

    async def provision_personal_workspace(
        self,
        *,
        auth_subject: str,
        workspace_name: str,
    ) -> ProvisionedPersonalWorkspace:
        del workspace_name
        self.subjects.append(auth_subject)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _provisioned_identity() -> ProvisionedPersonalWorkspace:
    return ProvisionedPersonalWorkspace(
        user_id=uuid4(),
        workspace_id=uuid4(),
        membership_id=uuid4(),
        kind=WorkspaceKind.PERSONAL,
        role=WorkspaceRole.ADMIN,
    )


async def test_actor_provider_maps_exact_verified_subject_through_provisioning() -> None:
    identity = _provisioned_identity()
    verifier = _ActorVerifier(VerifiedAuthPrincipal(subject="verified-subject"))
    store = _ActorProvisioningStore(identity)
    provider = SupabaseActorProvider(verifier, ProvisioningService(store))

    actor = await provider.get_actor("exact-access-token")

    assert actor == ActorContext(user_id=identity.user_id, subject="verified-subject")
    assert verifier.tokens == ["exact-access-token"]
    assert store.subjects == ["verified-subject"]


async def test_actor_provider_does_not_provision_after_verification_failure() -> None:
    error = AccessTokenVerificationError(TokenVerificationReason.INVALID_SIGNATURE)
    verifier = _ActorVerifier(error)
    store = _ActorProvisioningStore(_provisioned_identity())
    provider = SupabaseActorProvider(verifier, ProvisioningService(store))

    with pytest.raises(AccessTokenVerificationError) as captured:
        await provider.get_actor("invalid-token")

    assert captured.value is error
    assert store.subjects == []


async def test_actor_provider_preserves_provisioning_failure() -> None:
    error = RuntimeError("database unavailable")
    verifier = _ActorVerifier(VerifiedAuthPrincipal(subject="verified-subject"))
    store = _ActorProvisioningStore(error)
    provider = SupabaseActorProvider(verifier, ProvisioningService(store))

    with pytest.raises(RuntimeError, match="database unavailable") as captured:
        await provider.get_actor("valid-token")

    assert captured.value is error
    assert verifier.tokens == ["valid-token"]
    assert store.subjects == ["verified-subject"]


async def test_nbf_is_optional(key_a: _KeyMaterial) -> None:
    claims = _claims()
    del claims["nbf"]

    principal = await _verifier(_LocalPyJWKClient([_jwks(key_a)])).verify_access_token(
        _token(key_a, claims=claims)
    )

    assert principal.subject == "supabase-user-subject"


async def test_wrong_signature_is_rejected(
    key_a: _KeyMaterial,
    key_other: _KeyMaterial,
) -> None:
    token = _token(key_other, headers={"kid": key_a.kid})
    verifier = _verifier(_LocalPyJWKClient([_jwks(key_a)]))

    await _assert_invalid(verifier, token, TokenVerificationReason.INVALID_SIGNATURE)


@pytest.mark.parametrize(
    ("claim", "value", "reason"),
    [
        ("iss", "https://attacker.invalid/auth/v1", TokenVerificationReason.INVALID_ISSUER),
        ("aud", "service_role", TokenVerificationReason.INVALID_AUDIENCE),
    ],
)
async def test_wrong_trusted_claim_is_rejected(
    key_a: _KeyMaterial,
    claim: str,
    value: str,
    reason: TokenVerificationReason,
) -> None:
    verifier = _verifier(_LocalPyJWKClient([_jwks(key_a)]))

    await _assert_invalid(verifier, _token(key_a, claims=_claims(**{claim: value})), reason)


async def test_expired_beyond_leeway_is_rejected(key_a: _KeyMaterial) -> None:
    verifier = _verifier(_LocalPyJWKClient([_jwks(key_a)]))
    token = _token(
        key_a,
        claims=_claims(exp=datetime.now(UTC) - timedelta(seconds=90)),
    )

    await _assert_invalid(verifier, token, TokenVerificationReason.EXPIRED)


async def test_expiry_inside_fixed_clock_skew_is_accepted(key_a: _KeyMaterial) -> None:
    client = _LocalPyJWKClient([_jwks(key_a)])
    token = _token(
        key_a,
        claims=_claims(exp=datetime.now(UTC) - timedelta(seconds=5)),
    )

    principal = await _verifier(client).verify_access_token(token)

    assert principal.subject == "supabase-user-subject"
    assert SUPABASE_JWT_LEEWAY_SECONDS == 30


async def test_future_nbf_beyond_leeway_is_rejected(key_a: _KeyMaterial) -> None:
    verifier = _verifier(_LocalPyJWKClient([_jwks(key_a)]))
    token = _token(
        key_a,
        claims=_claims(nbf=datetime.now(UTC) + timedelta(seconds=90)),
    )

    await _assert_invalid(verifier, token, TokenVerificationReason.NOT_YET_VALID)


async def test_missing_subject_is_rejected(key_a: _KeyMaterial) -> None:
    claims = _claims()
    del claims["sub"]
    verifier = _verifier(_LocalPyJWKClient([_jwks(key_a)]))

    await _assert_invalid(
        verifier,
        _token(key_a, claims=claims),
        TokenVerificationReason.MISSING_SUBJECT,
    )


@pytest.mark.parametrize("claim", ["iss", "aud", "exp"])
async def test_other_required_claims_cannot_be_omitted(
    key_a: _KeyMaterial,
    claim: str,
) -> None:
    claims = _claims()
    del claims[claim]

    await _assert_invalid(
        _verifier(_LocalPyJWKClient([_jwks(key_a)])),
        _token(key_a, claims=claims),
        TokenVerificationReason.MALFORMED,
    )


@pytest.mark.parametrize("subject", ["   ", 123])
async def test_blank_or_non_string_subject_is_rejected(
    key_a: _KeyMaterial,
    subject: object,
) -> None:
    verifier = _verifier(_LocalPyJWKClient([_jwks(key_a)]))

    await _assert_invalid(
        verifier,
        _token(key_a, claims=_claims(sub=subject)),
        TokenVerificationReason.INVALID_SUBJECT,
    )


async def test_missing_kid_rejects_without_fetching_any_key(key_a: _KeyMaterial) -> None:
    client = _LocalPyJWKClient([_jwks(key_a)])
    token = jwt.encode(_claims(), key_a.private_key, algorithm="ES256")

    await _assert_invalid(
        _verifier(client),
        token,
        TokenVerificationReason.MISSING_KID,
    )
    assert client.fetch_count == 0


@pytest.mark.parametrize("algorithm", ["none", "HS256"])
async def test_none_and_hs256_are_rejected_before_jwks_lookup(
    key_a: _KeyMaterial,
    algorithm: str,
) -> None:
    client = _LocalPyJWKClient([_jwks(key_a)])
    key: str | None = None if algorithm == "none" else "attacker-shared-secret-at-least-32-bytes"
    token = jwt.encode(
        _claims(),
        key,
        algorithm=algorithm,
        headers={"kid": key_a.kid},
    )

    await _assert_invalid(
        _verifier(client),
        token,
        TokenVerificationReason.UNSUPPORTED_ALGORITHM,
    )
    assert client.fetch_count == 0


async def test_token_header_cannot_switch_fixed_algorithm_allowlist(
    key_a: _KeyMaterial,
) -> None:
    client = _LocalPyJWKClient([_jwks(key_a)])
    switched = _replace_algorithm_header(_token(key_a), "HS256")

    await _assert_invalid(
        _verifier(client),
        switched,
        TokenVerificationReason.UNSUPPORTED_ALGORITHM,
    )
    assert SUPABASE_JWT_ALGORITHMS == ("ES256",)
    assert client.fetch_count == 0


async def test_known_kid_uses_unexpired_jwks_set_cache(
    key_a: _KeyMaterial,
) -> None:
    client = _LocalPyJWKClient([_jwks(key_a)])
    verifier = _verifier(client)

    await verifier.verify_access_token(_token(key_a))
    await verifier.verify_access_token(_token(key_a))

    assert client.fetch_count == 1
    assert client.jwk_set_cache is not None
    assert client.jwk_set_cache.lifespan == 300
    assert client.timeout == 5
    assert not hasattr(client.get_signing_key, "cache_info")


async def test_unknown_kid_refreshes_once_and_accepts_rotated_key(
    key_a: _KeyMaterial,
    key_b: _KeyMaterial,
) -> None:
    client = _LocalPyJWKClient([_jwks(key_a), _jwks(key_a, key_b)])
    verifier = _verifier(client)
    await verifier.verify_access_token(_token(key_a))

    principal = await verifier.verify_access_token(_token(key_b))

    assert principal.subject == "supabase-user-subject"
    assert client.fetch_count == 2


async def test_unknown_kid_remaining_absent_refreshes_once_then_fails(
    key_a: _KeyMaterial,
    key_b: _KeyMaterial,
) -> None:
    client = _LocalPyJWKClient([_jwks(key_a), _jwks(key_a)])
    verifier = _verifier(client)
    await verifier.verify_access_token(_token(key_a))

    await _assert_invalid(
        verifier,
        _token(key_b),
        TokenVerificationReason.UNKNOWN_KID,
    )
    assert client.fetch_count == 2


async def test_warm_trusted_cache_verifies_known_key_during_outage(
    key_a: _KeyMaterial,
) -> None:
    outage = PyJWKClientConnectionError("local JWKS unavailable")
    client = _LocalPyJWKClient([_jwks(key_a), outage])
    verifier = _verifier(client)
    await verifier.verify_access_token(_token(key_a))

    principal = await verifier.verify_access_token(_token(key_a))

    assert principal.subject == "supabase-user-subject"
    assert client.fetch_count == 1


async def test_expired_cache_is_not_used_during_outage(
    key_a: _KeyMaterial,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _LocalPyJWKClient([_jwks(key_a), PyJWKClientConnectionError("local JWKS unavailable")])
    verifier = _verifier(client)
    await verifier.verify_access_token(_token(key_a))
    assert client.jwk_set_cache is not None
    cached_at = client.jwk_set_cache.jwk_set_with_timestamp
    assert cached_at is not None
    monkeypatch.setattr(
        jwt.jwk_set_cache.time,
        "monotonic",
        lambda: cached_at.get_timestamp() + SUPABASE_JWKS_TTL_SECONDS + 1,
    )

    with pytest.raises(JwksUnavailableError):
        await verifier.verify_access_token(_token(key_a))

    assert client.fetch_count == 2


async def test_initial_jwks_outage_fails_closed_without_retry_or_principal(
    key_a: _KeyMaterial,
) -> None:
    client = _LocalPyJWKClient([PyJWKClientConnectionError("local JWKS unavailable")])

    with pytest.raises(JwksUnavailableError) as captured:
        await _verifier(client).verify_access_token(_token(key_a))

    assert captured.value.reason is TokenVerificationReason.JWKS_UNAVAILABLE
    assert client.fetch_count == 1


async def test_unknown_kid_refresh_outage_fails_closed(
    key_a: _KeyMaterial,
    key_b: _KeyMaterial,
) -> None:
    client = _LocalPyJWKClient([_jwks(key_a), PyJWKClientConnectionError("local JWKS unavailable")])
    verifier = _verifier(client)
    await verifier.verify_access_token(_token(key_a))

    with pytest.raises(JwksUnavailableError):
        await verifier.verify_access_token(_token(key_b))

    assert client.fetch_count == 2


async def test_observable_failure_surface_excludes_token_claims_and_subject(
    key_a: _KeyMaterial,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    subject_canary = "SUBJECT-CANARY-DO-NOT-LEAK"
    claim_canary = "CLAIM-CANARY-DO-NOT-LEAK"
    token = _token(
        key_a,
        claims=_claims(
            sub=subject_canary,
            aud=claim_canary,
            user_metadata={"canary": claim_canary},
        ),
    )

    error = await _assert_invalid(
        _verifier(_LocalPyJWKClient([_jwks(key_a)])),
        token,
        TokenVerificationReason.INVALID_AUDIENCE,
    )

    captured_output = capsys.readouterr()
    surfaces = "\n".join(
        [
            str(error),
            repr(error),
            caplog.text,
            captured_output.out,
            captured_output.err,
        ]
    )
    for canary in (token, subject_canary, claim_canary):
        assert canary not in surfaces
