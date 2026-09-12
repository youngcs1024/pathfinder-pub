from uuid import uuid4

import pytest

from app.auth.contracts import ActorContext
from app.auth.fake import FAKE_ACTOR_SUBJECT, FakeActorProvider
from app.domain.provisioning import (
    ProvisionedPersonalWorkspace,
    WorkspaceKind,
    WorkspaceRole,
)


class _ProvisioningService:
    def __init__(self) -> None:
        self.user_id = uuid4()
        self.subjects: list[str] = []

    async def provision_personal_workspace(
        self,
        subject: str,
    ) -> ProvisionedPersonalWorkspace:
        self.subjects.append(subject)
        return ProvisionedPersonalWorkspace(
            user_id=self.user_id,
            workspace_id=uuid4(),
            membership_id=uuid4(),
            kind=WorkspaceKind.PERSONAL,
            role=WorkspaceRole.ADMIN,
        )


async def test_fake_actor_uses_fixed_server_subject_and_formal_provisioning() -> None:
    provisioning = _ProvisioningService()
    provider = FakeActorProvider(provisioning)  # type: ignore[arg-type]

    first = await provider.get_actor()
    second = await provider.get_actor()

    assert (
        first
        == second
        == ActorContext(
            user_id=provisioning.user_id,
            subject=FAKE_ACTOR_SUBJECT,
        )
    )
    assert provisioning.subjects == [FAKE_ACTOR_SUBJECT, FAKE_ACTOR_SUBJECT]
    assert repr(first).startswith("<app.auth.contracts.ActorContext object at ")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("user_id", "not-a-uuid", "actor user_id must be a UUID"),
        ("subject", "  ", "actor subject must be a non-blank string"),
    ],
)
def test_actor_context_rejects_untrusted_shapes(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {"user_id": uuid4(), "subject": "subject"}
    values[field] = value
    with pytest.raises(TypeError, match=message):
        ActorContext(**values)  # type: ignore[arg-type]
