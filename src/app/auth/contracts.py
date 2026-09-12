from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True, slots=True, repr=False)
class ActorContext:
    user_id: UUID
    subject: str

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, UUID):
            raise TypeError("actor user_id must be a UUID")
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise TypeError("actor subject must be a non-blank string")


class ActorProvider(Protocol):
    async def get_actor(self, access_token: str | None = None) -> ActorContext: ...


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedAuthPrincipal:
    subject: str

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise TypeError("verified subject must be a non-blank string")


class AccessTokenVerifier(Protocol):
    async def verify_access_token(self, token: str) -> VerifiedAuthPrincipal: ...
