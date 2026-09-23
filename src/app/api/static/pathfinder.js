"use strict";

const state = {
  config: null,
  accessToken: null,
  refreshToken: null,
  me: null,
  workspace: null,
  run: null,
  action: null,
  streamController: null,
  streamGeneration: 0,
  lastEventId: 0,
  terminal: false,
  eventsUrl: null,
  contextGeneration: 0,
  submission: null,
  materialAliases: [],
  materialProjects: [],
  materialSources: [],
  materialImports: [],
  materialViewedImport: null,
  materialProjectId: null,
  materialSubmission: null,
  materialProjectSubmission: null,
  materialSourceSubmission: null,
  materialPollTimer: null,
};

const byId = (id) => document.getElementById(id);
const problemPanel = byId("problem-panel");
const loginPanel = byId("login-panel");
const appPanel = byId("app-panel");
const runPanel = byId("run-panel");
const timelinePanel = byId("timeline-panel");
const actionPanel = byId("action-panel");
const loginForm = byId("login-form");
const runForm = byId("run-form");

let runCreateInFlight = false;

function invalidateSubmission() {
  state.contextGeneration += 1;
  state.submission = null;
  runCreateInFlight = false;
  renderSubmissionControls();
}

function isCurrentSubmission(submission) {
  return state.submission === submission
    && submission.generation === state.contextGeneration
    && submission.actorId === state.me?.user_id
    && submission.workspaceId === state.workspace?.workspace_id;
}

function renderSubmissionControls() {
  const submission = state.submission;
  byId("create-run").disabled = true;
  byId("retry-submission").hidden = !submission || Boolean(submission.accepted);
  byId("retry-submission").disabled = true;
  byId("retry-run-read").hidden = !submission?.accepted || submission.readComplete;
  byId("retry-run-read").disabled = runCreateInFlight;
  byId("submission-status").textContent = !submission ? "No submission retained in this page."
    : submission.accepted ? `Accepted Run ID: ${submission.accepted.run_id}. ${submission.readComplete ? "Current state loaded." : "Current state must still be read."}`
    : runCreateInFlight ? "Submitting the fixed request snapshot…"
    : submission.conflict ? "Request conflict. Stop retrying this submission; review before explicitly creating another task."
    : "The original request may already have created a task. Retry this submission uses its original inputs, even if the form has changed.";
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined && text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}

function sectionTitle(text) {
  return node("h3", text);
}

function clear(element) {
  element.replaceChildren();
}

function addFact(list, label, value) {
  list.append(node("dt", label), node("dd", value ?? "—"));
}

function jsonBlock(value) {
  return node("pre", JSON.stringify(value, null, 2));
}

function safeExternalHttpUrl(value) {
  try {
    const parsed = new URL(String(value));
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
  } catch (_error) {
    return null;
  }
}

function retryGuidance(operation, status) {
  if (status === 409 && operation === "createRun") return "The request key conflicts with an existing submission. No new key will be generated automatically.";
  if (status === 422) return "Correct the submitted fields, then submit a new request.";
  if (status === 404) return "Stop and verify the current workspace/resource. The server does not reveal whether an inaccessible resource exists.";
  if (status === 409 && operation === "decision") return "The approval changed. Refresh the action review; do not blindly repeat a mutation.";
  const guidance = {
    me: "This identity/workspace read is safe to retry manually. A 401 gets at most one session refresh.",
    readRun: "This authoritative run read is safe to retry manually.",
    readAction: "This authoritative action review is safe to retry manually.",
    stream: "Reconnect safely with the exact last acknowledged event ID. Invalid/future cursors fail closed.",
    createRun: "Use Retry this submission to resend the retained key and original inputs. Creating another task is a separate intent and may create a second run.",
    cancel: "The exact same cancellation request is safe to retry; cancellation converges idempotently.",
    decision: "After response loss, only the exact same actor, decision, expected version, and reason may be retried. A conflict requires a state refresh.",
    login: "Check the existing account credentials and try again. No Pathfinder session was created.",
  };
  return guidance[operation] || "Review the operation state before retrying.";
}

function showProblem(problem, operation) {
  clear(problemPanel);
  const title = problem.title || "Request failed";
  problemPanel.append(node("h2", problem.status ? `${title} (${problem.status})` : title));
  const facts = node("dl");
  addFact(facts, "Detail", problem.detail || "The response did not include more detail.");
  addFact(facts, "Type", problem.type || "about:blank");
  addFact(facts, "Request ID", problem.request_id || "unavailable");
  addFact(facts, "Retry guidance", retryGuidance(operation, problem.status));
  problemPanel.append(facts);
  problemPanel.hidden = false;
}

function hideProblem() {
  problemPanel.hidden = true;
  clear(problemPanel);
}

function reportInterfaceFailure(error) {
  showProblem(
    error.problem || {
      title: "Interface error",
      detail: error.message || String(error),
    },
    "ui",
  );
}

class PathfinderProblem extends Error {
  constructor(problem) {
    super(problem.detail || problem.title || "Request failed");
    this.problem = problem;
  }
}

async function problemFromResponse(response) {
  const fallback = {
    type: "about:blank",
    title: "Request failed",
    status: response.status,
    detail: `HTTP ${response.status}`,
    request_id: response.headers.get("x-request-id"),
  };
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/problem+json")) return fallback;
  try {
    return Object.assign(fallback, await response.json());
  } catch (_error) {
    return fallback;
  }
}

function clearSession() {
  state.accessToken = null;
  state.refreshToken = null;
}

async function refreshSession(signal) {
  const generation = state.contextGeneration;
  if (!state.refreshToken || !state.config || state.config.auth_mode !== "supabase") return false;
  const response = await fetch(`${state.config.supabase_url}/auth/v1/token?grant_type=refresh_token`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      apikey: state.config.supabase_publishable_key,
    },
    body: JSON.stringify({ refresh_token: state.refreshToken }),
    signal,
  });
  if (generation !== state.contextGeneration) throw new DOMException("Context changed", "AbortError");
  if (!response.ok) {
    clearSession();
    return false;
  }
  const session = await response.json();
  if (generation !== state.contextGeneration) throw new DOMException("Context changed", "AbortError");
  if (!session.access_token || !session.refresh_token) {
    clearSession();
    return false;
  }
  state.accessToken = session.access_token;
  state.refreshToken = session.refresh_token;
  return true;
}

async function apiFetch(path, options = {}, refreshed = false, generation = state.contextGeneration) {
  if (generation !== state.contextGeneration) throw new DOMException("Context changed", "AbortError");
  const headers = new Headers(options.headers || {});
  headers.set("Accept", options.accept || "application/json");
  if (options.body !== undefined) headers.set("Content-Type", "application/json");
  if (state.config.auth_mode === "supabase" && state.accessToken) {
    headers.set("Authorization", `Bearer ${state.accessToken}`);
  }
  const response = await fetch(path, {
    method: options.method || "GET",
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
  });
  if (generation !== state.contextGeneration) throw new DOMException("Context changed", "AbortError");
  if (response.status === 401 && state.config.auth_mode === "supabase") {
    if (!refreshed && await refreshSession(options.signal)) return apiFetch(path, options, true, generation);
    logout(true);
  }
  if (!response.ok) throw new PathfinderProblem(await problemFromResponse(response));
  if (response.status === 204) return null;
  return response.json();
}

async function login(email, password) {
  const response = await fetch(`${state.config.supabase_url}/auth/v1/token?grant_type=password`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      apikey: state.config.supabase_publishable_key,
    },
    body: JSON.stringify({ email, password }),
  });
  if (!response.ok) {
    throw new PathfinderProblem({
      type: "about:blank",
      title: "Sign-in failed",
      status: response.status,
      detail: "The supplied credentials were not accepted.",
      request_id: response.headers.get("x-request-id"),
    });
  }
  const session = await response.json();
  if (!session.access_token || !session.refresh_token) throw new Error("Supabase returned no session");
  state.accessToken = session.access_token;
  state.refreshToken = session.refresh_token;
}

function abortStream() {
  state.streamGeneration += 1;
  if (state.streamController) state.streamController.abort();
  state.streamController = null;
}

function resetProjection() {
  abortStream();
  state.run = null;
  state.action = null;
  state.lastEventId = 0;
  state.terminal = false;
  state.eventsUrl = null;
  byId("retry-events").hidden = true;
  clear(runPanel);
  clear(timelinePanel);
  clear(actionPanel);
  runPanel.hidden = true;
  timelinePanel.hidden = true;
  actionPanel.hidden = true;
}

function logout(showLogin) {
  resetMaterials();
  invalidateSubmission();
  abortStream();
  clearSession();
  state.me = null;
  state.workspace = null;
  resetProjection();
  appPanel.hidden = true;
  clear(byId("session-actions"));
  if (showLogin && state.config && state.config.auth_mode === "supabase") loginPanel.hidden = false;
}

function renderSessionActions() {
  const container = byId("session-actions");
  clear(container);
  if (state.config.auth_mode === "supabase" && state.me) {
    const button = node("button", "Log out", "secondary");
    button.type = "button";
    button.addEventListener("click", () => logout(true));
    container.append(button);
  }
}

async function loadMe() {
  const generation = state.contextGeneration;
  try {
    const me = await apiFetch("/api/v1/me");
    if (generation !== state.contextGeneration) return;
    if (state.me?.user_id !== me.user_id) {
      invalidateSubmission();
      resetProjection();
    }
    state.me = me;
    loginPanel.hidden = true;
    appPanel.hidden = false;
    renderSessionActions();
    renderWorkspaces(me.workspaces || []);
    loadMaterials().catch(reportInterfaceFailure);
    hideProblem();
  } catch (error) {
    if (generation !== state.contextGeneration) return;
    showProblem(error.problem || { title: "Workspace discovery failed", detail: error.message }, "me");
  }
}

function renderWorkspaces(workspaces) {
  const select = byId("workspace-select");
  clear(select);
  for (const workspace of workspaces) {
    const option = node("option", `${workspace.name} · ${workspace.kind} · ${workspace.role}`);
    option.value = workspace.workspace_id;
    select.append(option);
  }
  const selected = workspaces.find((item) => item.kind === "personal") || workspaces[0] || null;
  if (state.workspace?.workspace_id !== selected?.workspace_id) {
    invalidateSubmission();
    resetProjection();
    resetMaterials();
  }
  state.workspace = selected;
  if (selected) select.value = selected.workspace_id;
  renderWorkspaceMeta();
  select.disabled = workspaces.length === 0;
}

function renderWorkspaceMeta() {
  byId("workspace-meta").textContent = state.workspace
    ? `${state.workspace.name} · ${state.workspace.kind} · role: ${state.workspace.role}`
    : "No active workspace is available.";
}

function appendTimeline(eventType, data) {
  timelinePanel.hidden = false;
  let list = timelinePanel.querySelector("ol");
  if (!list) {
    timelinePanel.append(node("h2", "Event timeline"));
    list = node("ol", null, "timeline");
    timelinePanel.append(list);
  }
  const item = node("li");
  item.append(node("strong", `${data.seq} · ${eventType}`));
  if (data.occurred_at) item.append(node("time", data.occurred_at));
  list.append(item);
}

function renderClaims(parent, title, claims) {
  parent.append(sectionTitle(title));
  if (!claims || claims.length === 0) {
    parent.append(node("p", "None", "muted"));
    return;
  }
  for (const claim of claims) {
    const container = node("article", null, "claim");
    container.append(node("p", claim.text));
    const citations = (claim.citations || []).map((item) => `${item.source_id} / ${item.evidence_id}`);
    container.append(node("p", `Citations: ${citations.join(", ") || "none"}`, "muted"));
    parent.append(container);
  }
}

function renderUsageBucket(parent, label, bucket) {
  parent.append(node("h4", label));
  const facts = node("dl", null, "facts");
  addFact(facts, "Attempts / succeeded", `${bucket.attempt_count} / ${bucket.succeeded_count}`);
  addFact(facts, "Input tokens", bucket.input_tokens);
  addFact(facts, "Output tokens", bucket.output_tokens);
  addFact(facts, "Reasoning tokens", bucket.reasoning_output_tokens);
  addFact(facts, "Cached / cache-write", `${bucket.cached_input_tokens} / ${bucket.cache_write_input_tokens}`);
  const cost = bucket.cost_available && bucket.estimated_cost !== null
    ? `${bucket.estimated_cost} ${bucket.currency || ""}`.trim()
    : "unknown/unavailable";
  addFact(facts, "Estimated cost", cost);
  parent.append(facts);
}

function renderRun(detail) {
  state.run = Object.assign({}, state.run || {}, detail);
  clear(runPanel);
  runPanel.hidden = false;
  runPanel.append(node("h2", "Run"), node("span", detail.status, "status"));
  const facts = node("dl", null, "facts");
  addFact(facts, "Run ID", detail.run_id);
  addFact(facts, "Mode", detail.mode);
  addFact(facts, "Resume document", detail.resume_document_id);
  addFact(facts, "Error category", detail.error_category);
  runPanel.append(facts);

  if (detail.result) renderReport(runPanel, detail.result);
  runPanel.append(sectionTitle("Usage / cost"));
  renderUsageBucket(runPanel, "Chat", detail.usage.chat);
  renderUsageBucket(runPanel, "Embedding", detail.usage.embedding);
}

function renderReport(parent, result) {
  parent.append(sectionTitle(`Report · schema v${result.schema_version}`));
  renderClaims(parent, "Summary", result.summary);
  renderClaims(parent, "Findings", result.findings);
  parent.append(sectionTitle("Limitations"));
  for (const limitation of result.limitations || []) {
    parent.append(node("p", `${limitation.code}: ${limitation.detail}`));
  }
  if ((result.limitations || []).length === 0) parent.append(node("p", "None", "muted"));
  if (result.application_draft) renderClaims(parent, "Application draft", result.application_draft.paragraphs);
  parent.append(sectionTitle("Sources"));
  for (const source of result.sources || []) {
    const item = node("article", null, "source");
    item.append(node("strong", source.title));
    if ((source.source_type || "web") === "web") {
      if (source.url) {
        const safeUrl = safeExternalHttpUrl(source.url);
        if (safeUrl) {
          const link = node("a", source.url);
          link.href = safeUrl;
          link.rel = "noreferrer noopener";
          link.target = "_blank";
          item.append(node("br"), link);
        } else {
          item.append(node("p", source.url, "muted"));
        }
      }
      item.append(node("p", source.snippet || "", "muted"));
      if (source.published_at) item.append(node("p", `Published: ${source.published_at}`, "muted"));
    } else {
      item.append(node("p", `Workspace document: ${source.source_name} · ${source.document_id}`, "muted"));
    }
    parent.append(item);
  }
  parent.append(sectionTitle("Evidence"));
  for (const evidence of result.evidence || []) {
    const item = node("article", null, "evidence");
    item.append(node("p", evidence.text));
    const identity = evidence.source_type === "workspace_document"
      ? `document ${evidence.document_id}, chunk ${evidence.chunk_id}, ordinal ${evidence.ordinal}`
      : `source ${evidence.source_id}`;
    item.append(node("p", `${evidence.evidence_id} · ${identity}`, "muted"));
    parent.append(item);
  }
}

async function readProjection(path, signal) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  let timedOut = false;
  if (signal?.aborted) abort();
  signal?.addEventListener("abort", abort, { once: true });
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 30000);
  try {
    return await apiFetch(path, { signal: controller.signal });
  } catch (error) {
    if (timedOut && !signal?.aborted) throw new DOMException("Detail read timed out", "TimeoutError");
    throw error;
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}

async function fetchRun(operation = "readRun", { propagate = false, signal } = {}) {
  if (!state.workspace || !state.run) {
    if (propagate) throw new Error("Run projection unavailable");
    return;
  }
  const generation = state.streamGeneration;
  try {
    const detail = await readProjection(`/api/v1/workspaces/${state.workspace.workspace_id}/runs/${state.run.run_id}`, signal);
    if (generation !== state.streamGeneration) return;
    renderRun(detail);
    hideProblem();
  } catch (error) {
    if (generation !== state.streamGeneration) return;
    if (propagate) throw error;
    showProblem(error.problem || { title: "Run read failed", detail: error.message }, operation);
  }
}

async function cancelCurrentRun() {
  showProblem({ status: 410, title: "Historical workflow retired", detail: "Historical runs are read-only. Resume creation is not available yet." }, "readRun");
}

function renderAction(review) {
  state.action = review;
  clear(actionPanel);
  actionPanel.hidden = false;
  actionPanel.append(node("h2", "Action review"), node("span", review.status, "status"));
  const facts = node("dl", null, "facts");
  addFact(facts, "Intent ID", review.action_intent_id);
  addFact(facts, "Action", `${review.action_key} · revision ${review.action_revision}`);
  addFact(facts, "Tool / effect", `${review.tool_name} / ${review.effect}`);
  addFact(facts, "Args canonicalization", `v${review.canonicalization_version}`);
  addFact(facts, "Args digest", review.args_digest);
  addFact(facts, "Target canonicalization", `v${review.target_canonicalization_version}`);
  addFact(facts, "Target digest", review.target_digest);
  addFact(facts, "Binding", `v${review.approval_binding_version} · ${review.approval_binding_digest}`);
  addFact(facts, "Approval", `${review.approval_request.status} · v${review.approval_request.version}`);
  addFact(facts, "Expires", review.approval_request.expires_at);
  addFact(facts, "Approval args digest", review.approval_request.args_digest);
  addFact(facts, "Approval target digest", review.approval_request.target_digest);
  addFact(facts, "Approval binding", `v${review.approval_request.approval_binding_version} · ${review.approval_request.approval_binding_digest}`);
  addFact(facts, "Policy version", review.approval_request.policy_version);
  addFact(facts, "Recovery attempts", review.recovery_attempts);
  actionPanel.append(facts);
  actionPanel.append(sectionTitle("Exact args snapshot"), jsonBlock(review.args_snapshot));
  actionPanel.append(sectionTitle("Exact target snapshot"), jsonBlock(review.target_snapshot));
  actionPanel.append(sectionTitle("Policy snapshot"), jsonBlock(review.approval_request.policy_snapshot));
  if (review.decision) actionPanel.append(sectionTitle("Decision"), jsonBlock(review.decision));
  if (review.result) actionPanel.append(sectionTitle("Action result"), jsonBlock(review.result));
  if (review.evidence) actionPanel.append(sectionTitle("Action evidence"), jsonBlock(review.evidence));
  if (review.manual_review_required || review.status === "outcome_unknown") {
    actionPanel.append(node("p", review.manual_review_instruction || "Verify the external target; do not retry the effect.", "manual-review"));
  }

}

async function fetchAction(actionIntentId, operation = "readAction", { propagate = false, signal } = {}) {
  if (!state.workspace) {
    if (propagate) throw new Error("Action projection unavailable");
    return;
  }
  const generation = state.streamGeneration;
  try {
    const review = await readProjection(`/api/v1/workspaces/${state.workspace.workspace_id}/action-intents/${actionIntentId}`, signal);
    if (generation !== state.streamGeneration) return;
    renderAction(review);
    hideProblem();
  } catch (error) {
    if (generation !== state.streamGeneration) return;
    if (propagate) throw error;
    showProblem(error.problem || { title: "Action review failed", detail: error.message }, operation);
  }
}

async function submitDecision(decision, reason) {
  showProblem({ status: 410, title: "Historical workflow retired", detail: "Historical runs are read-only. Resume creation is not available yet." }, "readRun");
}

async function applyEvent(eventType, data) {
  const options = { propagate: true, signal: state.streamController?.signal };
  if (eventType === "action.proposed" && data.payload && data.payload.action_intent_id) {
    await fetchAction(data.payload.action_intent_id, "readAction", options);
  } else if (eventType.startsWith("action.") || eventType.startsWith("approval.")) {
    if (state.action) await fetchAction(state.action.action_intent_id, "readAction", options);
  }
  const terminal = ["run.completed", "run.failed", "run.cancelled"].includes(eventType);
  if (terminal || eventType === "run.status_changed" || eventType === "report.completed") {
    await fetchRun("readRun", options);
  }
  return terminal;
}

async function parseSseFrame(frame, generation) {
  if (generation !== state.streamGeneration) return;
  if (!frame || frame.startsWith(":")) return;
  let id = null;
  let eventType = "message";
  let dataText = "";
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("id:")) id = Number(line.slice(3).trim());
    else if (line.startsWith("event:")) eventType = line.slice(6).trim();
    else if (line.startsWith("data:")) dataText += line.slice(5).trimStart();
  }
  if (!Number.isSafeInteger(id) || !dataText) throw new Error("Incomplete SSE event frame");
  if (id <= state.lastEventId) return;
  const data = JSON.parse(dataText);
  const terminal = await applyEvent(eventType, data);
  if (generation !== state.streamGeneration) return;
  appendTimeline(eventType, data);
  state.lastEventId = id;
  if (terminal) state.terminal = true;
}

async function consumeSse(response, generation) {
  if (!response.body) throw new Error("Streaming response body unavailable");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (generation === state.streamGeneration) {
      const chunk = await reader.read();
      if (generation !== state.streamGeneration) return;
      buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
      let boundary = buffer.search(/\r?\n\r?\n/);
      while (boundary >= 0) {
        const match = buffer.slice(boundary).match(/^\r?\n\r?\n/);
        const separatorLength = match ? match[0].length : 2;
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + separatorLength);
        await parseSseFrame(frame, generation);
        if (generation !== state.streamGeneration || state.terminal) return;
        boundary = buffer.search(/\r?\n\r?\n/);
      }
      if (chunk.done) return;
    }
  } finally {
    try {
      await reader.cancel();
    } catch (_error) {
      // An aborted/errored stream can reject cancellation; preserve the original failure.
    } finally {
      reader.releaseLock();
    }
  }
}

function waitForReconnect(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const abort = () => {
      window.clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, milliseconds);
    signal.addEventListener("abort", abort, { once: true });
  });
}

function offerEventRecovery() {
  byId("retry-events").hidden = !state.eventsUrl || !state.run || state.terminal;
}

async function recoverEvents() {
  if (!state.workspace || !state.run || !state.eventsUrl || state.terminal) return;
  await streamEvents(state.eventsUrl);
}

async function streamEvents(eventsUrl) {
  abortStream();
  state.eventsUrl = eventsUrl;
  byId("retry-events").hidden = true;
  const controller = new AbortController();
  state.streamController = controller;
  const generation = state.streamGeneration;
  const delays = [500, 1000, 2000, 4000, 5000];
  let attempt = 0;
  try {
    while (!state.terminal && generation === state.streamGeneration) {
      const previousCursor = state.lastEventId;
      try {
        const headers = new Headers({
          Accept: "text/event-stream",
          "Last-Event-ID": String(state.lastEventId),
        });
        if (state.config.auth_mode === "supabase" && state.accessToken) {
          headers.set("Authorization", `Bearer ${state.accessToken}`);
        }
        let response = await fetch(eventsUrl, { headers, signal: controller.signal });
        if (generation !== state.streamGeneration) return;
        if (response.status === 401 && state.config.auth_mode === "supabase") {
          const refreshed = await refreshSession(controller.signal);
          if (generation !== state.streamGeneration) return;
          if (!refreshed) {
            logout(true);
            return;
          }
          headers.set("Authorization", `Bearer ${state.accessToken}`);
          response = await fetch(eventsUrl, { headers, signal: controller.signal });
        }
        if (generation !== state.streamGeneration) return;
        if (!response.ok) throw new PathfinderProblem(await problemFromResponse(response));
        await consumeSse(response, generation);
        if (generation !== state.streamGeneration) return;
        if (state.terminal) return;
      } catch (error) {
        if (controller.signal.aborted || generation !== state.streamGeneration) return;
        showProblem(error.problem || {
          type: "about:blank",
          title: "Event stream disconnected",
          status: 0,
          detail: error.message,
          request_id: null,
        }, "stream");
        const status = error.problem?.status;
        if (status >= 400 && status < 500 && ![408, 429].includes(status)) {
          offerEventRecovery();
          return;
        }
      }
      if (state.lastEventId > previousCursor) attempt = 0;
      if (attempt >= delays.length) {
        showProblem({
          title: "Event recovery paused",
          detail: "No events could be processed after five reconnects. Resume event reads to continue this task from its last processed event.",
        }, "stream");
        offerEventRecovery();
        return;
      }
      const delay = delays[attempt];
      attempt += 1;
      try {
        await waitForReconnect(delay, controller.signal);
      } catch (error) {
        if (error.name === "AbortError") return;
        throw error;
      }
    }
  } finally {
    if (state.streamController === controller) state.streamController = null;
  }
}

async function createRun() {
  showProblem({ status: 410, title: "Historical workflow retired", detail: "Historical runs are read-only. Resume creation is not available yet." }, "readRun");
}

async function retrySubmission() {
  showProblem({ status: 410, title: "Historical workflow retired", detail: "Historical runs are read-only. Resume creation is not available yet." }, "readRun");
}

async function submitRunIntent(submission) {
  showProblem({ status: 410, title: "Historical workflow retired", detail: "Historical runs are read-only. Resume creation is not available yet." }, "readRun");
}

async function readAcceptedRun(submission = state.submission) {
  if (runCreateInFlight || !submission?.accepted || !isCurrentSubmission(submission)) return;
  runCreateInFlight = true;
  renderSubmissionControls();
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 30000);
  try {
    const detail = await apiFetch(`/api/v1/workspaces/${submission.workspaceId}/runs/${submission.accepted.run_id}`, { signal: controller.signal });
    if (!isCurrentSubmission(submission)) return;
    if (detail?.run_id !== submission.accepted.run_id) throw new Error("Run identity mismatch");
    renderRun(detail);
    submission.readComplete = true;
    hideProblem();
  } catch (error) {
    if (!isCurrentSubmission(submission)) return;
    showProblem(error.problem || { title: "Run read failed", detail: "The task was accepted. Retry reading its current state; do not submit it again." }, "readRun");
    return;
  } finally {
    window.clearTimeout(timer);
    if (isCurrentSubmission(submission)) {
      runCreateInFlight = false;
      renderSubmissionControls();
    }
  }
  // Replay from zero even for terminal runs to recover the timeline and action review.
  if (isCurrentSubmission(submission)) {
    streamEvents(submission.accepted.events_url).catch((error) => {
      if (isCurrentSubmission(submission)) reportInterfaceFailure(error);
    });
  }
}

function updateResumeRequirement() {
  const application = byId("run-mode").value === "application";
  byId("resume-document-id").required = application;
  byId("resume-guidance").textContent = application
    ? "Required for application. Paste an immutable document ID produced by the ingestion CLI."
    : "Optional for research.";
}

function resetMaterials() {
  if (state.materialPollTimer !== null) window.clearTimeout(state.materialPollTimer);
  state.materialPollTimer = null;
  state.materialAliases = [];
  state.materialProjects = [];
  state.materialSources = [];
  state.materialImports = [];
  state.materialViewedImport = null;
  state.materialProjectId = null;
  state.materialSubmission = null;
  state.materialProjectSubmission = null;
  state.materialSourceSubmission = null;
  clear(byId("material-project-select"));
  clear(byId("material-alias-select"));
  clear(byId("material-source-list"));
  clear(byId("material-import-history"));
  clear(byId("material-import-progress"));
  byId("material-import-status").textContent = "";
  byId("material-retry-import").hidden = true;
  byId("material-refresh-import").hidden = true;
  byId("material-import").disabled = true;
}

function materialBase() {
  return `/api/v2/workspaces/${state.workspace.workspace_id}`;
}

function renderMaterials() {
  const projectSelect = byId("material-project-select");
  clear(projectSelect);
  for (const project of state.materialProjects) {
    const option = node("option", project.name);
    option.value = project.id;
    projectSelect.append(option);
  }
  if (state.materialProjectId) projectSelect.value = state.materialProjectId;
  projectSelect.disabled = !state.materialProjects.length;
  const aliasSelect = byId("material-alias-select");
  clear(aliasSelect);
  for (const alias of state.materialAliases) {
    const option = node("option", `${alias.name} · ${alias.kind}`);
    option.value = alias.name;
    aliasSelect.append(option);
  }
  byId("material-add-source").disabled = !state.materialProjectId || !state.materialAliases.length;
  byId("materials-availability").textContent = state.materialAliases.length
    ? "Select authorized sources. Imports read fixed snapshots in the background."
    : "No material aliases are configured. Ask the operator to register a private allowlist before importing.";
  const list = byId("material-source-list");
  clear(list);
  for (const source of state.materialSources) {
    const label = node("label");
    const box = node("input");
    box.type = "checkbox";
    box.id = `material-source-${source.id}`;
    box.checked = true;
    label.append(box, node("span", `${source.alias} · ${source.kind}`));
    list.append(label);
  }
  byId("material-import").disabled = !state.materialSources.length || Boolean(state.materialSubmission);
  const history = byId("material-import-history");
  clear(history);
  for (const item of state.materialImports) {
    const option = node("option", `${item.created_at} · ${item.status} · ${item.id}`);
    option.value = item.id;
    history.append(option);
  }
  byId("material-view-import").disabled = !state.materialImports.length;
}

async function loadMaterials() {
  if (!state.workspace) return;
  const generation = state.contextGeneration;
  const workspaceId = state.workspace.workspace_id;
  const [aliases, projects] = await Promise.all([
    apiFetch(`${materialBase()}/material-aliases`),
    apiFetch(`${materialBase()}/projects`),
  ]);
  if (generation !== state.contextGeneration || state.workspace?.workspace_id !== workspaceId) return;
  state.materialAliases = aliases;
  state.materialProjects = projects;
  if (!projects.some(item => item.id === state.materialProjectId)) {
    state.materialProjectId = projects[0]?.id || null;
  }
  state.materialSources = state.materialProjectId
    ? await apiFetch(`${materialBase()}/projects/${state.materialProjectId}/material-sources`)
    : [];
  state.materialImports = state.materialProjectId
    ? await apiFetch(`${materialBase()}/projects/${state.materialProjectId}/imports`)
    : [];
  if (generation !== state.contextGeneration || state.workspace?.workspace_id !== workspaceId) return;
  renderMaterials();
}

async function createMaterialProject() {
  const name = byId("material-project-name").value;
  const pending = state.materialProjectSubmission?.name === name
    ? state.materialProjectSubmission : { name, key: crypto.randomUUID() };
  state.materialProjectSubmission = pending;
  const project = await apiFetch(`${materialBase()}/projects`, {
    method: "POST", headers: { "Idempotency-Key": pending.key }, body: { name },
  });
  state.materialProjectSubmission = null;
  state.materialProjectId = project.id;
  byId("material-project-form").reset();
  await loadMaterials();
}

async function addMaterialSource() {
  const projectId = state.materialProjectId;
  if (!projectId) return;
  const alias = byId("material-alias-select").value;
  const pending = state.materialSourceSubmission?.projectId === projectId
    && state.materialSourceSubmission?.alias === alias
    ? state.materialSourceSubmission : { projectId, alias, key: crypto.randomUUID() };
  state.materialSourceSubmission = pending;
  await apiFetch(`${materialBase()}/projects/${projectId}/material-sources`, {
    method: "POST", headers: { "Idempotency-Key": pending.key }, body: { alias },
  });
  state.materialSourceSubmission = null;
  await loadMaterials();
}

function renderMaterialProgress(progress) {
  const container = byId("material-import-progress");
  clear(container);
  const summary = node("p", `Import ${progress.status} · ${progress.snapshots.length} source snapshots`);
  container.append(summary);
  if (progress.error_category) container.append(node("p", `Failure: ${progress.error_category}`));
  for (const snapshot of progress.snapshots) {
    const item = node("article", null, "source");
    item.append(node("strong", `${snapshot.source_id} · ${snapshot.source_revision}`));
    item.append(node("p", `${snapshot.indexed_count}/${snapshot.file_count} files indexed`));
    if (snapshot.unindexed_paths.length) {
      item.append(node("p", `${snapshot.unindexed_paths.length} files not indexed`, "muted"));
    }
    container.append(item);
  }
}

async function refreshMaterialImport() {
  const submission = state.materialViewedImport || state.materialSubmission;
  if (!submission?.accepted || state.workspace?.workspace_id !== submission.workspaceId) return;
  if (state.materialPollTimer !== null) window.clearTimeout(state.materialPollTimer);
  state.materialPollTimer = null;
  const progress = await apiFetch(
    `/api/v2/workspaces/${submission.workspaceId}/projects/${submission.projectId}/imports/${submission.accepted.import_id}`,
  );
  if (state.materialViewedImport !== submission && state.materialSubmission !== submission) return;
  renderMaterialProgress(progress);
  byId("material-import-status").textContent = `Import ${progress.status}.`;
  byId("material-refresh-import").hidden = false;
  if (["queued", "running"].includes(progress.status)) {
    state.materialPollTimer = window.setTimeout(() => {
      refreshMaterialImport().catch(reportInterfaceFailure);
    }, 2000);
  } else if (state.materialSubmission === submission) {
    state.materialViewedImport = submission;
    state.materialSubmission = null;
    byId("material-import").disabled = !state.materialSources.length;
    state.materialImports.unshift({
      id: submission.accepted.import_id,
      status: progress.status,
      created_at: new Date().toISOString(),
    });
    renderMaterials();
  }
}

async function sendMaterialImport(submission) {
  if (state.materialSubmission !== submission || state.workspace?.workspace_id !== submission.workspaceId) return;
  try {
    const accepted = await apiFetch(
      `/api/v2/workspaces/${submission.workspaceId}/projects/${submission.projectId}/imports`,
      {
        method: "POST",
        headers: { "Idempotency-Key": submission.key },
        body: submission.body,
      },
    );
    if (state.materialSubmission !== submission) return;
    submission.accepted = accepted;
    state.materialViewedImport = submission;
    byId("material-retry-import").hidden = true;
    byId("material-import-status").textContent = `Accepted import ${accepted.import_id}.`;
    await refreshMaterialImport();
  } catch (error) {
    if (state.materialSubmission !== submission) return;
    byId("material-retry-import").hidden = error.problem?.status === 409;
    byId("material-import-status").textContent = "The response was not confirmed. Retry sends the same key and source selection.";
    throw error;
  }
}

async function submitMaterialImport() {
  if (!state.workspace || !state.materialProjectId || state.materialSubmission) return;
  const sourceIds = state.materialSources
    .filter(item => byId(`material-source-${item.id}`).checked)
    .map(item => item.id);
  if (!sourceIds.length) return;
  const submission = {
    workspaceId: state.workspace.workspace_id,
    projectId: state.materialProjectId,
    key: crypto.randomUUID(),
    body: { source_ids: sourceIds },
    accepted: null,
  };
  state.materialSubmission = submission;
  state.materialViewedImport = submission;
  byId("material-import").disabled = true;
  await sendMaterialImport(submission);
}

async function bootstrap() {
  try {
    const response = await fetch("/api/v1/ui-config", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new PathfinderProblem(await problemFromResponse(response));
    state.config = await response.json();
    if (state.config.auth_mode === "fake") await loadMe();
    else loginPanel.hidden = false;
  } catch (error) {
    showProblem(error.problem || { title: "UI bootstrap failed", detail: error.message }, "me");
  }
}

async function submitLogin() {
  const email = byId("email").value;
  const password = byId("password").value;

  try {
    await login(email, password);
  } catch (error) {
    showProblem(error.problem || { title: "Sign-in failed", detail: error.message }, "login");
    return;
  }

  loginForm.reset();
  await loadMe();
}

loginForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitLogin().catch(reportInterfaceFailure);
});

byId("workspace-select").addEventListener("change", (event) => {
  const selected = (state.me.workspaces || []).find((item) => item.workspace_id === event.target.value);
  invalidateSubmission();
  resetProjection();
  resetMaterials();
  state.workspace = selected || null;
  renderWorkspaceMeta();
  hideProblem();
  loadMaterials().catch(reportInterfaceFailure);
});

byId("run-mode").addEventListener("change", updateResumeRequirement);
runForm.addEventListener("submit", (event) => {
  event.preventDefault();
  createRun().catch(reportInterfaceFailure);
});
byId("retry-submission").addEventListener("click", () => {
  retrySubmission().catch(reportInterfaceFailure);
});
byId("retry-run-read").addEventListener("click", () => {
  readAcceptedRun().catch(reportInterfaceFailure);
});
byId("retry-events").addEventListener("click", () => {
  recoverEvents().catch(reportInterfaceFailure);
});
byId("material-project-form").addEventListener("submit", (event) => {
  event.preventDefault();
  createMaterialProject().catch(reportInterfaceFailure);
});
byId("material-project-select").addEventListener("change", (event) => {
  if (state.materialPollTimer !== null) window.clearTimeout(state.materialPollTimer);
  state.materialPollTimer = null;
  state.materialSubmission = null;
  state.materialViewedImport = null;
  state.materialProjectId = event.target.value;
  clear(byId("material-import-progress"));
  byId("material-retry-import").hidden = true;
  byId("material-refresh-import").hidden = true;
  loadMaterials().catch(reportInterfaceFailure);
});
byId("material-source-form").addEventListener("submit", (event) => {
  event.preventDefault();
  addMaterialSource().catch(reportInterfaceFailure);
});
byId("material-import").addEventListener("click", () => {
  submitMaterialImport().catch(reportInterfaceFailure);
});
byId("material-retry-import").addEventListener("click", () => {
  if (state.materialSubmission) sendMaterialImport(state.materialSubmission).catch(reportInterfaceFailure);
});
byId("material-refresh-import").addEventListener("click", () => {
  refreshMaterialImport().catch(reportInterfaceFailure);
});
byId("material-view-import").addEventListener("click", () => {
  const importId = byId("material-import-history").value;
  if (!importId || !state.workspace || !state.materialProjectId) return;
  state.materialViewedImport = {
    workspaceId: state.workspace.workspace_id,
    projectId: state.materialProjectId,
    accepted: { import_id: importId },
  };
  refreshMaterialImport().catch(reportInterfaceFailure);
});
renderSubmissionControls();
updateResumeRequirement();
resetMaterials();
bootstrap();
