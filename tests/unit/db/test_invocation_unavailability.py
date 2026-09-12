from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from psycopg.errors import LockNotAvailable

from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.tenancy import SqlAlchemyTenantResolver
from app.db.tool_invocations import SqlAlchemyToolInvocationRecorder
from app.domain.errors import DomainUnavailableError
from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import ToolInvocationAuthorizationError
from app.llm.invocations import (
    LOCKED_CHAT_MODEL,
    LLMInvocationAttempt,
    LLMInvocationAuthorizationError,
)


@pytest.mark.parametrize("operation", ["llm", "tool", "tenant"])
@pytest.mark.parametrize("unavailable", [False, True])
async def test_database_guards_distinguish_failed_queries_from_missing_membership(
    operation, unavailable
):
    events = []

    class Session:
        @asynccontextmanager
        async def begin(self):
            try:
                yield
            finally:
                events.append("transaction_exited")

        async def scalar(self, statement):
            if unavailable:
                raise LockNotAvailable("SQL-PARAM-CANARY")
            return None

    @asynccontextmanager
    async def factory():
        try:
            yield Session()
        finally:
            events.append("session_closed")

    workspace, actor, run = uuid4(), uuid4(), uuid4()
    if operation == "llm":
        call = SqlAlchemyInvocationRecorder(factory).prepare(
            LLMInvocationAttempt(
                invocation_id=uuid4(),
                workspace_id=workspace,
                actor_user_id=actor,
                run_id=run,
                invocation_kind="chat",
                provider="fake",
                model=LOCKED_CHAT_MODEL,
                graph_node="plan",
                prompt_version=f"sha256:{'a' * 64}",
                request_hash=f"sha256:{'b' * 64}",
            )
        )
        denied = LLMInvocationAuthorizationError
    elif operation == "tool":
        call = SqlAlchemyToolInvocationRecorder(factory).reserve(
            invocation_id=uuid4(),
            workspace_id=workspace,
            actor_user_id=actor,
            run_id=run,
            tool_name="web_search",
            effect=ToolEffect.READ_ONLY,
            args_digest=f"sha256:{'b' * 64}",
            call_limit=3,
        )
        denied = ToolInvocationAuthorizationError
    else:
        call = SqlAlchemyTenantResolver(factory).resolve_tenant(
            workspace_id=workspace,
            actor_user_id=actor,
        )
        denied = None
    if unavailable or denied is not None:
        with pytest.raises(DomainUnavailableError if unavailable else denied):
            await call
    else:
        assert await call is None
    assert events[-1] == "session_closed"
    if operation != "tenant":
        assert events == ["transaction_exited", "session_closed"]


async def test_execution_permission_query_failure_is_unavailable_after_close():
    from app.db.run_execution import SqlAlchemyRunExecutionReader
    from app.domain.runs import CURRENT_GRAPH_VERSION

    closed = []

    class Session:
        async def execute(self, statement):
            raise LockNotAvailable("GUARD-SQL-CANARY")

    @asynccontextmanager
    async def factory():
        try:
            yield Session()
        finally:
            closed.append(True)

    with pytest.raises(DomainUnavailableError):
        await SqlAlchemyRunExecutionReader(factory).assert_execution_allowed(
            run_id=uuid4(),
            workspace_id=uuid4(),
            actor_user_id=uuid4(),
            graph_version=CURRENT_GRAPH_VERSION,
        )
    assert closed == [True]
