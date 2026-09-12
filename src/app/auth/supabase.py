from __future__ import annotations

import asyncio

import jwt
from jwt import PyJWKClient
from jwt.exceptions import (
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAlgorithmError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidSignatureError,
    InvalidSubjectError,
    InvalidTokenError,
    MissingRequiredClaimError,
    PyJWKClientConnectionError,
    PyJWKClientError,
    PyJWTError,
)

from app.auth.contracts import AccessTokenVerifier, ActorContext, VerifiedAuthPrincipal
from app.auth.errors import (
    AccessTokenVerificationError,
    JwksUnavailableError,
    TokenVerificationReason,
)
from app.config import Settings
from app.domain.provisioning import ProvisioningService

SUPABASE_JWT_ALGORITHMS = ("ES256",)
SUPABASE_JWKS_TTL_SECONDS = 300
SUPABASE_JWKS_HTTP_TIMEOUT_SECONDS = 5
SUPABASE_JWT_LEEWAY_SECONDS = 30
_REQUIRED_CLAIMS = ("iss", "aud", "exp", "sub")


class SupabaseJwtVerifier:
    def __init__(
        self,
        settings: Settings,
        *,
        jwk_client: PyJWKClient | None = None,
    ) -> None:
        issuer = settings.supabase_issuer
        jwks_url = settings.supabase_jwks_url
        if settings.auth_mode != "supabase" or issuer is None or jwks_url is None:
            raise ValueError("Supabase JWT verifier requires trusted Supabase settings")

        self._issuer = issuer
        self._audience = settings.supabase_jwt_audience
        self._jwk_client = jwk_client or PyJWKClient(
            jwks_url,
            cache_jwk_set=True,
            lifespan=SUPABASE_JWKS_TTL_SECONDS,
            cache_keys=False,
            timeout=SUPABASE_JWKS_HTTP_TIMEOUT_SECONDS,
        )

    async def verify_access_token(self, token: str) -> VerifiedAuthPrincipal:
        if not isinstance(token, str) or not token.strip():
            raise AccessTokenVerificationError(TokenVerificationReason.MALFORMED)

        try:
            header = jwt.get_unverified_header(token)
        except PyJWTError:
            raise AccessTokenVerificationError(TokenVerificationReason.MALFORMED) from None

        algorithm = header.get("alg")
        if algorithm != SUPABASE_JWT_ALGORITHMS[0]:
            raise AccessTokenVerificationError(TokenVerificationReason.UNSUPPORTED_ALGORITHM)
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid.strip():
            raise AccessTokenVerificationError(TokenVerificationReason.MISSING_KID)

        try:
            signing_key = await asyncio.to_thread(
                self._jwk_client.get_signing_key_from_jwt,
                token,
            )
        except PyJWKClientConnectionError:
            raise JwksUnavailableError() from None
        except PyJWKClientError:
            raise AccessTokenVerificationError(TokenVerificationReason.UNKNOWN_KID) from None
        except PyJWTError:
            raise AccessTokenVerificationError(TokenVerificationReason.MALFORMED) from None

        try:
            claims = jwt.decode(
                token,
                key=signing_key.key,
                algorithms=list(SUPABASE_JWT_ALGORITHMS),
                audience=self._audience,
                issuer=self._issuer,
                leeway=SUPABASE_JWT_LEEWAY_SECONDS,
                options={
                    "require": list(_REQUIRED_CLAIMS),
                    "strict_aud": True,
                    "verify_iat": False,
                },
            )
        except ExpiredSignatureError:
            raise AccessTokenVerificationError(TokenVerificationReason.EXPIRED) from None
        except ImmatureSignatureError:
            raise AccessTokenVerificationError(TokenVerificationReason.NOT_YET_VALID) from None
        except InvalidIssuerError:
            raise AccessTokenVerificationError(TokenVerificationReason.INVALID_ISSUER) from None
        except InvalidAudienceError:
            raise AccessTokenVerificationError(TokenVerificationReason.INVALID_AUDIENCE) from None
        except MissingRequiredClaimError as error:
            reason = (
                TokenVerificationReason.MISSING_SUBJECT
                if error.claim == "sub"
                else TokenVerificationReason.MALFORMED
            )
            raise AccessTokenVerificationError(reason) from None
        except InvalidSubjectError:
            raise AccessTokenVerificationError(TokenVerificationReason.INVALID_SUBJECT) from None
        except InvalidSignatureError:
            raise AccessTokenVerificationError(TokenVerificationReason.INVALID_SIGNATURE) from None
        except InvalidAlgorithmError:
            raise AccessTokenVerificationError(
                TokenVerificationReason.UNSUPPORTED_ALGORITHM
            ) from None
        except InvalidTokenError:
            raise AccessTokenVerificationError(TokenVerificationReason.MALFORMED) from None

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            raise AccessTokenVerificationError(TokenVerificationReason.INVALID_SUBJECT)
        return VerifiedAuthPrincipal(subject=subject)


class SupabaseActorProvider:
    def __init__(
        self,
        verifier: AccessTokenVerifier,
        provisioning_service: ProvisioningService,
    ) -> None:
        self._verifier = verifier
        self._provisioning_service = provisioning_service

    async def get_actor(self, access_token: str | None = None) -> ActorContext:
        if access_token is None:
            raise AccessTokenVerificationError(TokenVerificationReason.MALFORMED)
        principal = await self._verifier.verify_access_token(access_token)
        provisioned = await self._provisioning_service.provision_personal_workspace(
            principal.subject
        )
        return ActorContext(
            user_id=provisioned.user_id,
            subject=principal.subject,
        )
