from uuid import uuid4

import pytest

from app.domain.errors import DomainValidationError
from app.domain.provisioning import (
    PERSONAL_WORKSPACE_NAME,
    ProvisionedPersonalWorkspace,
    ProvisioningService,
    WorkspaceKind,
    WorkspaceRole,
)


class _RecordingStore:
    def __init__(self, result: ProvisionedPersonalWorkspace) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def provision_personal_workspace(
        self,
        *,
        auth_subject: str,
        workspace_name: str,
    ) -> ProvisionedPersonalWorkspace:
        self.calls.append((auth_subject, workspace_name))
        return self.result


def _result() -> ProvisionedPersonalWorkspace:
    return ProvisionedPersonalWorkspace(
        user_id=uuid4(),
        workspace_id=uuid4(),
        membership_id=uuid4(),
        kind=WorkspaceKind.PERSONAL,
        role=WorkspaceRole.ADMIN,
    )


async def test_service_uses_fixed_personal_workspace_contract() -> None:
    expected = _result()
    store = _RecordingStore(expected)
    service = ProvisioningService(store)

    actual = await service.provision_personal_workspace("fake-user-123")

    assert actual is expected
    assert store.calls == [("fake-user-123", PERSONAL_WORKSPACE_NAME)]


@pytest.mark.parametrize("auth_subject", ["", " ", "\t\n"])
async def test_service_rejects_blank_auth_subject(auth_subject: str) -> None:
    store = _RecordingStore(_result())
    service = ProvisioningService(store)

    with pytest.raises(DomainValidationError, match="auth subject must not be blank"):
        await service.provision_personal_workspace(auth_subject)

    assert store.calls == []
