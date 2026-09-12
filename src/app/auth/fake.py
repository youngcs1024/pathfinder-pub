from app.auth.contracts import ActorContext
from app.domain.provisioning import ProvisioningService

FAKE_ACTOR_SUBJECT = "pathfinder-local-user"


class FakeActorProvider:
    def __init__(self, provisioning_service: ProvisioningService) -> None:
        self._provisioning_service = provisioning_service

    async def get_actor(self, access_token: str | None = None) -> ActorContext:
        del access_token
        provisioned = await self._provisioning_service.provision_personal_workspace(
            FAKE_ACTOR_SUBJECT
        )
        return ActorContext(
            user_id=provisioned.user_id,
            subject=FAKE_ACTOR_SUBJECT,
        )
