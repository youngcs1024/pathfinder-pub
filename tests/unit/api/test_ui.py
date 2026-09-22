import json
import re
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.main import create_app

_STATIC_DIR = Path(__file__).resolve().parents[3] / "src" / "app" / "api" / "static"
_PUBLISHABLE_KEY = "sb_publishable_test-public-key"


async def _get(application, path: str) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        return await client.get(path)


def _function_source(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


async def test_root_and_local_static_assets_are_public_and_served() -> None:
    application = create_app(Settings(log_level="ERROR"))

    root = await _get(application, "/")
    javascript = await _get(application, "/static/pathfinder.js")
    stylesheet = await _get(application, "/static/pathfinder.css")

    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert 'src="/static/pathfinder.js"' in root.text
    assert 'href="/static/pathfinder.css"' in root.text
    assert "http://" not in root.text
    assert "https://" not in root.text
    assert javascript.status_code == 200
    assert javascript.headers["content-type"].startswith(
        ("text/javascript", "application/javascript")
    )
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert "/" not in application.openapi()["paths"]
    assert all(not path.startswith("/static") for path in application.openapi()["paths"])


async def test_static_requests_do_not_resolve_actor_or_execute_product_services() -> None:
    application = create_app(Settings(log_level="ERROR"))
    sentinel = object()
    application.state.actor_provider = sentinel
    application.state.run_service = sentinel
    application.state.run_event_reader = sentinel
    application.state.approval_service = sentinel

    for path in ("/", "/static/pathfinder.js", "/static/pathfinder.css"):
        response = await _get(application, path)
        assert response.status_code == 200

    assert application.state.actor_provider is sentinel
    assert application.state.run_service is sentinel
    assert application.state.run_event_reader is sentinel
    assert application.state.approval_service is sentinel


async def test_fake_ui_config_has_no_browser_credential_and_needs_no_auth() -> None:
    application = create_app(Settings(log_level="ERROR"))

    response = await _get(application, "/api/v1/ui-config")

    assert response.status_code == 200
    assert response.json() == {
        "auth_mode": "fake",
        "supabase_url": None,
        "supabase_publishable_key": None,
    }


async def test_supabase_ui_config_exposes_only_trusted_browser_values() -> None:
    database_canary = "DB-CREDENTIAL-CANARY"
    provider_canary = "PROVIDER-SECRET-CANARY"
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

    response = await _get(application, "/api/v1/ui-config")

    assert response.status_code == 200
    assert response.json() == {
        "auth_mode": "supabase",
        "supabase_url": "https://abcdefghijklmnopqrst.supabase.co",
        "supabase_publishable_key": _PUBLISHABLE_KEY,
    }
    assert database_canary not in response.text
    assert provider_canary not in response.text
    assert "service_role" not in response.text
    assert "sb_secret_" not in response.text
    assert "access_token" not in response.text
    assert "refresh_token" not in response.text
    assert "jwks" not in response.text.lower()


def test_packaged_static_files_exist_and_javascript_uses_safe_browser_contract() -> None:
    html_path = _STATIC_DIR / "index.html"
    javascript_path = _STATIC_DIR / "pathfinder.js"
    stylesheet_path = _STATIC_DIR / "pathfinder.css"

    assert html_path.is_file()
    assert javascript_path.is_file()
    assert stylesheet_path.is_file()
    source = javascript_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")

    for required in ("fetch", "Authorization", "Last-Event-ID", "AbortController"):
        assert required in source
    for forbidden in (
        "EventSource",
        "localStorage",
        "sessionStorage",
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "document.writeln",
        "eval(",
        "new Function",
        "srcdoc",
        "console.",
    ):
        assert forbidden not in source
    assert "access_token=" not in source
    assert "refresh_token=" not in source
    assert '<script src="http' not in html
    assert re.findall(r"<script\b[^>]*>", html) == ['<script src="/static/pathfinder.js" defer>']
    assert re.findall(r"<link\b[^>]*rel=\"stylesheet\"[^>]*>", html) == [
        '<link rel="stylesheet" href="/static/pathfinder.css">'
    ]
    assert re.search(r"\son[a-z]+\s*=", html, flags=re.IGNORECASE) is None


def test_untrusted_ui_fields_have_only_text_rendering_paths() -> None:
    source = (_STATIC_DIR / "pathfinder.js").read_text(encoding="utf-8")

    assert "element.textContent = String(text)" in source
    assert 'return node("pre", JSON.stringify(value, null, 2))' in source
    for rendering_path in (
        'node("p", claim.text)',
        'node("strong", source.title)',
        'node("p", source.snippet || "", "muted")',
        "source.source_name",
        'node("p", evidence.text)',
        'addFact(facts, "Detail", problem.detail',
        "jsonBlock(review.args_snapshot)",
        "jsonBlock(review.target_snapshot)",
        "jsonBlock(review.result)",
    ):
        assert rendering_path in source


@pytest.mark.parametrize(
    "payload",
    (
        "<script>window.__pathfinder_xss = 1</script>",
        '<img src=x onerror="window.__pathfinder_xss = 1">',
        '</pre><svg onload="window.__pathfinder_xss = 1">',
    ),
)
def test_representative_xss_payloads_remain_data_for_text_only_renderer(payload: str) -> None:
    source = (_STATIC_DIR / "pathfinder.js").read_text(encoding="utf-8")

    assert json.loads(json.dumps({"untrusted": payload}))["untrusted"] == payload
    assert "element.textContent = String(text)" in source
    assert 'return node("pre", JSON.stringify(value, null, 2))' in source
    for executable_sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "srcdoc"):
        assert executable_sink not in source


def test_external_links_are_created_only_from_http_protocol_guard() -> None:
    source = (_STATIC_DIR / "pathfinder.js").read_text(encoding="utf-8")

    assert "function safeExternalHttpUrl(value)" in source
    assert 'parsed.protocol === "http:" || parsed.protocol === "https:"' in source
    assert "const safeUrl = safeExternalHttpUrl(source.url)" in source
    assert "link.href = safeUrl" in source
    assert "link.href = source.url" not in source
    assert 'link.rel = "noreferrer noopener"' in source
    assert 'link.target = "_blank"' in source


def test_login_submit_keeps_dom_event_outside_async_lifetime() -> None:
    source = (_STATIC_DIR / "pathfinder.js").read_text(encoding="utf-8")
    listener = """loginForm.addEventListener(\"submit\", (event) => {
  event.preventDefault();
  submitLogin().catch(reportInterfaceFailure);
});"""

    assert 'const loginForm = byId("login-form")' in source
    assert "async function submitLogin()" in source
    assert listener in source
    assert "event.currentTarget" not in source
    assert 'loginForm.addEventListener("submit", async' not in source


def test_historical_run_submit_guards_create_post_and_separates_stream_stage() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "tests/fixtures/legacy_ui/pathfinder.js"
    ).read_text(encoding="utf-8")
    create_run = _function_source(
        source,
        "async function createRun()",
        "async function retrySubmission",
    )
    submit = _function_source(
        source,
        "async function submitRunIntent",
        "async function readAcceptedRun",
    )
    read = _function_source(
        source,
        "async function readAcceptedRun",
        "function updateResumeRequirement",
    )

    assert 'const runForm = byId("run-form")' in source
    assert 'runForm.addEventListener("submit", (event) => {' in source
    assert "createRun().catch(reportInterfaceFailure)" in source
    assert "if (runCreateInFlight) return" in create_run
    assert "if (runCreateInFlight) return" in submit
    assert "runCreateInFlight = true" in submit
    assert "runCreateInFlight = false" in submit
    assert submit.index("} finally {") < submit.index("await readAcceptedRun(submission)")
    assert "streamEvents(" not in submit
    assert read.index("await apiFetch(") < read.index("renderRun(detail)")
    assert read.index("renderRun(detail)") < read.index("streamEvents(")
    assert read.index("} finally {") < read.index("streamEvents(")
    assert "renderRunAccepted" not in source
    assert 'node("span", accepted.status' not in source


def test_historical_submission_snapshot_and_key_are_generated_only_for_explicit_new_task() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "tests/fixtures/legacy_ui/pathfinder.js"
    ).read_text(encoding="utf-8")
    create_run = _function_source(
        source,
        "async function createRun()",
        "async function retrySubmission",
    )
    retry = _function_source(
        source,
        "async function retrySubmission",
        "async function submitRunIntent",
    )
    submit = _function_source(
        source,
        "async function submitRunIntent",
        "async function readAcceptedRun",
    )

    assert "const payload = Object.freeze({" in create_run
    assert 'query: byId("run-query").value' in create_run
    assert "actorId: state.me.user_id" in create_run
    assert "workspaceId: state.workspace.workspace_id" in create_run
    assert 'typeof globalThis.crypto?.randomUUID !== "function"' in create_run
    assert source.count("globalThis.crypto.randomUUID()") == 1
    assert "Math.random" not in source
    assert "!state.submission.accepted" in create_run
    assert create_run.index("window.confirm(") < create_run.index("globalThis.crypto.randomUUID()")
    assert "await submitRunIntent(submission)" in retry
    assert "byId(" not in retry + submit
    assert 'headers: { "Idempotency-Key": submission.requestId }' in submit
    assert "body: submission.payload" in submit
    assert "${submission.workspaceId}/runs" in submit


def test_historical_uncertain_submission_retains_identity_and_conflict_stops_retry() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "tests/fixtures/legacy_ui/pathfinder.js"
    ).read_text(encoding="utf-8")
    submit = _function_source(
        source,
        "async function submitRunIntent",
        "async function readAcceptedRun",
    )
    retry = _function_source(
        source,
        "async function retrySubmission",
        "async function submitRunIntent",
    )

    assert "controller.abort(), 30000" in submit
    assert "signal: controller.signal" in submit
    assert "window.clearTimeout(timer)" in submit
    assert "!uuid4.test(accepted.run_id)" in submit
    assert "accepted.events_url !==" in submit
    assert "submission.accepted = Object.freeze(accepted)" in submit
    assert "submission.conflict = error.problem?.status === 409" in submit
    assert "submission.conflict" in retry
    assert "state.submission = null" not in submit
    assert "requestId:" not in submit
    assert "while (" not in submit
    assert "setInterval" not in submit
    assert "No new key will be generated automatically" in source


def test_accepted_read_retry_never_posts_or_discards_the_receipt() -> None:
    source = (_STATIC_DIR / "pathfinder.js").read_text(encoding="utf-8")
    read = _function_source(
        source,
        "async function readAcceptedRun",
        "function updateResumeRequirement",
    )

    assert "!submission?.accepted" in read
    assert "${submission.accepted.run_id}" in read
    assert "detail?.run_id !== submission.accepted.run_id" in read
    assert "submission.readComplete = true" in read
    assert "submission.accepted =" not in read
    assert 'method: "POST"' not in read
    assert "submitRunIntent(" not in read
    assert "readAcceptedRun().catch(reportInterfaceFailure)" in source
    assert "if (isCurrentSubmission(submission))" in read
    assert "if (!isCurrentSubmission(submission)) return" in read


def test_submission_buttons_and_memory_only_recovery_limit_are_explicit() -> None:
    source = (_STATIC_DIR / "pathfinder.js").read_text(encoding="utf-8")
    html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    for button_id in ("create-run", "retry-submission", "retry-run-read"):
        assert f'id="{button_id}"' in html
        assert f'byId("{button_id}").disabled = ' in source
    for button_id in ("retry-submission", "retry-run-read"):
        assert f'<button id="{button_id}" type="button" hidden' in html
    assert 'id="submission-status" class="muted" aria-live="polite"' in html
    assert "recovery after refresh is not automatic" in html
    assert "The original task may already exist" in html
    assert "Retry uses the original inputs" in html
    assert "sessionStorage" not in source
    assert "localStorage" not in source


def test_historical_context_changes_invalidate_submission_and_ignore_late_results() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "tests/fixtures/legacy_ui/pathfinder.js"
    ).read_text(encoding="utf-8")
    invalidate = _function_source(
        source,
        "function invalidateSubmission",
        "function isCurrentSubmission",
    )
    current = _function_source(
        source,
        "function isCurrentSubmission",
        "function renderSubmissionControls",
    )
    logout = _function_source(source, "function logout", "function renderSessionActions")
    me = _function_source(source, "async function loadMe", "function renderWorkspaces")
    workspaces = _function_source(
        source,
        "function renderWorkspaces",
        "function renderWorkspaceMeta",
    )
    change = source[source.index('byId("workspace-select").addEventListener') :]

    assert "state.contextGeneration += 1" in invalidate
    assert "state.submission = null" in invalidate
    assert "runCreateInFlight = false" in invalidate
    for binding in (
        "state.submission === submission",
        "state.contextGeneration",
        "state.me?.user_id",
        "state.workspace?.workspace_id",
    ):
        assert binding in current
    for boundary in (logout, me, workspaces, change):
        assert "invalidateSubmission()" in boundary
    for start, end in (
        ("async function submitRunIntent", "async function readAcceptedRun"),
        ("async function readAcceptedRun", "function updateResumeRequirement"),
    ):
        body = _function_source(source, start, end)
        assert body.count("if (!isCurrentSubmission(submission)) return") >= 2
        assert "if (isCurrentSubmission(submission))" in body


def test_auth_retry_preserves_options_and_checks_context_before_replaying() -> None:
    source = (_STATIC_DIR / "pathfinder.js").read_text(encoding="utf-8")
    fetch = _function_source(source, "async function apiFetch", "async function login(")
    refresh = _function_source(source, "async function refreshSession", "async function apiFetch")

    assert "!refreshed && await refreshSession(options.signal)" in fetch
    assert "return apiFetch(path, options, true, generation)" in fetch
    assert fetch.count("generation !== state.contextGeneration") == 2
    assert refresh.index("generation !== state.contextGeneration") < refresh.index("clearSession()")
    assert refresh.rindex("generation !== state.contextGeneration") < refresh.index(
        "state.accessToken = session.access_token"
    )


def test_historical_old_sse_and_projection_reads_cannot_overwrite_new_submission() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "tests/fixtures/legacy_ui/pathfinder.js"
    ).read_text(encoding="utf-8")
    for start, end in (
        ("async function fetchRun", "async function cancelCurrentRun"),
        ("async function fetchAction", "async function submitDecision"),
        ("async function consumeSse", "function waitForReconnect"),
        ("async function streamEvents", "async function createRun"),
    ):
        body = _function_source(source, start, end)
        assert "if (generation !== state.streamGeneration) return" in body
    frame = _function_source(source, "async function parseSseFrame", "async function consumeSse")
    assert frame.index("await applyEvent") < frame.index("state.lastEventId = id")
    assert frame.index("await applyEvent") < frame.rindex("generation !== state.streamGeneration")


def test_historical_decision_failure_refresh_stays_bound_to_captured_workspace() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "tests/fixtures/legacy_ui/pathfinder.js"
    ).read_text(encoding="utf-8")
    submit_decision = _function_source(
        source,
        "async function submitDecision",
        "async function applyEvent",
    )

    assert "await loadMe()" not in submit_decision
    assert "const workspaceId = state.workspace.workspace_id" in submit_decision
    assert "const actionIntentId = state.action.action_intent_id" in submit_decision
    assert "decisionProblem.status !== 409" in submit_decision
    assert "`/api/v1/workspaces/${workspaceId}/action-intents/${actionIntentId}`" in submit_decision
    for binding in (
        "state.contextGeneration === contextGeneration",
        "state.streamGeneration === generation",
        "state.me?.user_id === actorId",
        "state.workspace?.workspace_id === workspaceId",
        "state.run?.run_id === runId",
        "state.action?.action_intent_id === actionIntentId",
    ):
        assert binding in submit_decision
    assert submit_decision.count("if (!current()) return") == 4
    assert "await fetchAction(actionIntentId)" not in submit_decision
    assert "submitDecision(decision, reason.value).catch(reportInterfaceFailure)" in source


def test_non_http_interface_problem_is_not_mislabeled_as_network_failure() -> None:
    source = (_STATIC_DIR / "pathfinder.js").read_text(encoding="utf-8")

    assert 'problem.status || "network"' not in source
    assert "problem.status ? `${title} (${problem.status})` : title" in source
    assert "function reportInterfaceFailure(error)" in source
