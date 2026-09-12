from enum import StrEnum


class TokenVerificationReason(StrEnum):
    MALFORMED = "malformed"
    UNSUPPORTED_ALGORITHM = "unsupported_algorithm"
    MISSING_KID = "missing_kid"
    UNKNOWN_KID = "unknown_kid"
    INVALID_SIGNATURE = "invalid_signature"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    INVALID_ISSUER = "invalid_issuer"
    INVALID_AUDIENCE = "invalid_audience"
    MISSING_SUBJECT = "missing_subject"
    INVALID_SUBJECT = "invalid_subject"
    JWKS_UNAVAILABLE = "jwks_unavailable"


class AccessTokenVerificationError(Exception):
    def __init__(self, reason: TokenVerificationReason) -> None:
        self.reason = reason
        super().__init__("access token is invalid")


class JwksUnavailableError(AccessTokenVerificationError):
    def __init__(self) -> None:
        super().__init__(TokenVerificationReason.JWKS_UNAVAILABLE)
        self.args = ("signing keys are temporarily unavailable",)
