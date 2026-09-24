"use strict";

const RESUME_JOB_MAX_BYTES = 32 * 1024;
const RESUME_TERMINAL = new Set(["completed", "failed", "cancelled"]);

function resumeBase() {
  return `/api/v2/workspaces/${state.workspace.workspace_id}/resume-sessions`;
}

function resumeLocationSelection() {
  const params = new URLSearchParams(window.location?.search || "");
  return { workspaceId: params.get("workspace_id"), sessionId: params.get("session_id") };
}

function rememberResumeSession(sessionId) {
  if (!window.history?.replaceState || !state.workspace) return;
  const params = new URLSearchParams(window.location?.search || "");
  params.set("workspace_id", state.workspace.workspace_id);
  if (sessionId) params.set("session_id", sessionId);
  else params.delete("session_id");
  window.history.replaceState(null, "", `/?${params.toString()}`);
}

function resetResumeGeneration() {
  state.resumeGeneration = (state.resumeGeneration || 0) + 1;
  state.resumeRequestController?.abort();
  state.resumeRequestController = new AbortController();
  state.resumeStreamController?.abort();
  state.resumeStreamController = null;
  state.resumeSessionId = null;
  state.resumeSession = null;
  state.resumeVersion = null;
  state.resumeArtifact = null;
  state.resumeQuestions = [];
  state.resumeUserFacts = [];
  state.resumeFeedbackHistory = [];
  state.resumeFeedbackSubmission = null;
  state.resumeSessions = [];
  state.resumeSubmission = null;
  state.resumeSubmitting = false;
  state.resumePreparing = false;
  state.resumeProfileVersionForForm = null;
  state.resumeLastEventId = 0;
  state.resumeTerminal = false;
  for (const id of ["resume-job-progress", "resume-job-questions", "resume-job-draft",
    "resume-job-coverage", "resume-job-delivery", "resume-job-history", "resume-job-projects"]) {
    clear(byId(id));
  }
  byId("resume-job-status").textContent = "";
  byId("resume-job-retry").hidden = true;
  byId("resume-job-new").hidden = true;
  byId("resume-job-file-preview").textContent = "";
  byId("resume-job-file").value = "";
  byId("resume-job-profile").textContent = "Loading profile…";
  byId("resume-job-submit").disabled = true;
}

function renderResumeGenerationSetup() {
  const selected = new Set(Array.from(byId("resume-job-projects").querySelectorAll("input:checked"))
    .map(item => item.value));
  const projects = byId("resume-job-projects");
  clear(projects);
  projects.append(node("legend", "Projects for this job"));
  for (const project of state.materialProjects || []) {
    const label = node("label");
    const input = node("input");
    input.type = "checkbox";
    input.value = project.id;
    input.checked = selected.has(project.id);
    label.append(input, node("span", project.name));
    projects.append(label);
  }
  const profile = state.resumeProfile;
  byId("resume-job-profile").textContent = profile
    ? `Current reviewed profile: version ${profile.version}. The session fixes this version and preference version ${profile.preference_version}.`
    : "Import and review a resume profile before creating a first draft.";
  byId("resume-job-submit").disabled = !profile || !state.materialProjects?.length
    || Boolean(state.resumeSubmission) || Boolean(state.resumeSubmitting) || Boolean(state.resumePreparing);
  if (profile && state.resumeProfileVersionForForm !== profile.version_id) {
    byId("resume-job-pages").value = String(profile.preferences.page_target);
    state.resumeProfileVersionForForm = profile.version_id;
  }
}

function renderResumeSessionList() {
  const select = byId("resume-job-history");
  clear(select);
  for (const item of state.resumeSessions) {
    const option = node("option", `${item.job_label} · ${item.run_status} · ${item.created_at}`);
    option.value = item.session_id;
    select.append(option);
  }
  if (state.resumeSessionId) select.value = state.resumeSessionId;
  select.disabled = !state.resumeSessions.length;
}

async function loadResumeSessions() {
  if (!state.workspace) return;
  const generation = state.contextGeneration;
  const resumeGeneration = state.resumeGeneration;
  const workspaceId = state.workspace.workspace_id;
  const sessions = await apiFetch(resumeBase(), { signal: state.resumeRequestController.signal });
  if (generation !== state.contextGeneration || state.workspace?.workspace_id !== workspaceId
      || resumeGeneration !== state.resumeGeneration) return;
  state.resumeSessions = sessions;
  renderResumeSessionList();
  const location = resumeLocationSelection();
  const selected = location.workspaceId === workspaceId && location.sessionId
    ? location.sessionId : state.resumeSessionId || sessions[0]?.session_id;
  if (selected) await openResumeSession(selected);
}

async function refreshResumeDetail(sessionId, generation) {
  const signal = state.resumeRequestController.signal;
  const detail = await apiFetch(`${resumeBase()}/${sessionId}`, { signal });
  if (detail.session_id !== sessionId) throw new Error("Session identity mismatch");
  let version = null;
  let artifact = null;
  if (detail.current_version_id) {
    version = await apiFetch(`${resumeBase()}/${sessionId}/versions/${detail.current_version_id}`, { signal });
    if (version.version_id !== detail.current_version_id || version.session_id !== sessionId) {
      throw new Error("Draft version identity mismatch");
    }
    artifact = await apiFetch(`/api/v2/workspaces/${state.workspace.workspace_id}/artifacts/${version.artifact_id}`, { signal });
    if (artifact.artifact_id !== version.artifact_id) throw new Error("Artifact identity mismatch");
  }
  let questions = [], userFacts = [], feedback = [];
  if (detail.current_version_id || (detail.initial_run_id && detail.run_id !== detail.initial_run_id)) {
    [questions, userFacts, feedback] = await Promise.all([
      apiFetch(`${resumeBase()}/${sessionId}/questions`, { signal }),
      apiFetch(`${resumeBase()}/${sessionId}/user-facts`, { signal }),
      apiFetch(`${resumeBase()}/${sessionId}/feedback`, { signal }),
    ]);
  }
  if (generation !== state.resumeGeneration || state.resumeSessionId !== sessionId) return;
  state.resumeQuestions = questions;
  state.resumeUserFacts = userFacts;
  state.resumeFeedbackHistory = feedback;
  state.resumeSession = detail;
  state.resumeVersion = version;
  state.resumeArtifact = artifact;
  state.resumeTerminal = RESUME_TERMINAL.has(detail.run_status);
  renderResumeSession();
}

async function openResumeSession(sessionId) {
  if (!state.workspace || !sessionId) return;
  state.resumeGeneration += 1;
  state.resumeStreamController?.abort();
  state.resumeStreamController = null;
  state.resumeSessionId = sessionId;
  state.resumeSession = null;
  state.resumeVersion = null;
  state.resumeArtifact = null;
  state.resumeQuestions = [];
  state.resumeUserFacts = [];
  state.resumeFeedbackHistory = [];
  state.resumeFeedbackSubmission = null;
  state.resumeLastEventId = 0;
  state.resumeTerminal = false;
  const generation = state.resumeGeneration;
  byId("resume-job-status").textContent = "Reading the saved job…";
  try {
    await refreshResumeDetail(sessionId, generation);
    if (generation !== state.resumeGeneration) return;
    rememberResumeSession(sessionId);
    renderResumeSessionList();
    byId("resume-job-status").textContent = "Saved job loaded.";
    if (!state.resumeTerminal) streamResumeEvents(sessionId, generation).catch(reportInterfaceFailure);
  } catch (error) {
    if (generation !== state.resumeGeneration) return;
    byId("resume-job-status").textContent = "Could not read this job. Refresh to retry.";
    showProblem(error.problem || { title: "Job read failed", detail: error.message }, "readRun");
  }
}

function displayResumeContent(target, content) {
  target.append(node("h3", "Draft content"), node("p", content.display_name));
  for (const item of content.education || []) {
    target.append(node("p", `${item.institution.text} · ${item.qualification.text} · ${item.period.text}`));
  }
  for (const project of content.projects || []) {
    const card = node("article", null, "source");
    card.append(node("h4", project.title.text), node("p", project.summary.text));
    for (const bullet of project.bullets || []) card.append(node("p", `• ${bullet.text}`));
    target.append(card);
  }
  for (const skill of content.skills || []) {
    target.append(node("p", `${skill.label}: ${skill.items.join(", ")}`));
  }
  target.append(node("p", "Content preview only. Check the actual layout after external compilation.", "muted"));
}

function coverageLabel(coverage) {
  if (!coverage) return "Not checked; no draft coverage was published.";
  const verification = {
    needs_human_review: "Needs human review",
    confirmed_gap: "Confirmed ability gap",
    material_insufficient: "Material is insufficient",
    unchecked: "Not checked",
  }[coverage.verification] || coverage.verification;
  const support = {
    supported: "Supported by selected facts",
    partial: "Partially supported",
    no_support_found: "No supporting fact selected",
  }[coverage.support] || coverage.support;
  return `${support} · ${verification}`;
}

function renderResumeSession() {
  const detail = state.resumeSession;
  if (!detail) return;
  const version = state.resumeVersion;
  const progress = byId("resume-job-progress");
  clear(progress);
  progress.append(node("h3", "Job progress"), node("p", `Run: ${detail.run_status}`));
  if (detail.error_category) progress.append(node("p", `Failure: ${detail.error_category}`));
  progress.append(node("p", `JD source: ${detail.job.source}${detail.job.filename ? ` · ${detail.job.filename}` : ""}`));
  progress.append(node("pre", detail.job.text));
  const projectNames = detail.project_ids.map(id =>
    state.materialProjects.find(project => project.id === id)?.name || id);
  progress.append(node("p", `Fixed profile ${detail.profile_version_id}; preference version ${detail.preference_version}; projects ${projectNames.join(", ")}`,
    "muted"));
  progress.append(node("p", `Run limits: ${detail.budget.max_model_calls} model calls, ${detail.budget.max_tool_calls} tool calls, ¥${detail.budget.max_cost_cny}. Page target: ${detail.override.page_target || "global preference"}.`, "muted"));
  if (!state.resumeTerminal) {
    const cancel = node("button", "Cancel this run", "danger");
    cancel.type = "button";
    cancel.addEventListener("click", () => cancelResumeSession().catch(reportInterfaceFailure));
    progress.append(cancel);
  }
  const questions = byId("resume-job-questions");
  clear(questions);
  const allQuestions = detail.result?.questions || version?.validation?.questions || [];
  questions.append(node("h3", "Questions and review"));
  for (const question of allQuestions) questions.append(node("p", question));
  if (!allQuestions.length) questions.append(node("p", "No open question was recorded for this run."));
  const draft = byId("resume-job-draft");
  clear(draft);
  if (version) displayResumeContent(draft, version.content);
  else if (detail.result?.outcome === "needs_input") {
    draft.append(node("h3", "No draft yet"), node("p", "More confirmed information is needed."));
  }
  const coverage = byId("resume-job-coverage");
  clear(coverage);
  coverage.append(node("h3", "JD requirements and evidence"));
  const coverageByRequirement = new Map((version?.coverage || []).map(item => [item.requirement_id, item]));
  const factsByVersion = new Map((version?.facts || []).map(item => [item.version_id, item]));
  const itemTextById = new Map();
  for (const project of version?.content?.projects || []) {
    (project.bullet_ids || []).forEach((id, index) => {
      itemTextById.set(id, project.bullets[index]?.text || id);
    });
  }
  for (const requirement of detail.requirements) {
    const card = node("article", null, "evidence");
    const relation = coverageByRequirement.get(requirement.id);
    card.append(node("p", `${requirement.kind}: “${requirement.quote}” (JD characters ${requirement.start}–${requirement.end})`));
    if (requirement.inference_basis) card.append(node("p", `Inference basis: ${requirement.inference_basis}`));
    card.append(node("p", coverageLabel(relation)));
    if (relation?.reason) card.append(node("p", `Selection: ${relation.reason}`));
    for (const itemId of relation?.item_ids || []) {
      card.append(node("p", `Draft line: ${itemTextById.get(itemId) || itemId}`));
    }
    for (const factId of relation?.fact_version_ids || []) {
      const fact = factsByVersion.get(factId);
      if (!fact) continue;
      card.append(node("p", `${fact.kind}: ${fact.claim}`));
      if (Object.keys(fact.conditions || {}).length) card.append(node("pre", JSON.stringify(fact.conditions, null, 2)));
      for (const evidence of fact.evidence || []) {
        card.append(node("p", `${evidence.path} · ${evidence.source_revision} · lines ${evidence.start_line}–${evidence.end_line}: ${evidence.quote}`));
      }
    }
    coverage.append(card);
  }
  if (!detail.requirements.length) coverage.append(node("p", "No JD requirement has been recorded yet."));
  const omissions = version?.validation?.omission_reasons || {};
  if (Object.keys(omissions).length) coverage.append(node("h4", "Confirmed facts omitted from this draft"));
  for (const [factId, reason] of Object.entries(omissions)) {
    const fact = factsByVersion.get(factId);
    coverage.append(node("p", `${fact?.claim || factId}: ${reason}`));
  }
  const delivery = byId("resume-job-delivery");
  clear(delivery);
  delivery.append(node("h3", "Delivery and confirmation"));
  delivery.append(node("p", version ? "Unconfirmed draft" : "No draft to confirm"));
  if (version && state.resumeArtifact) {
    delivery.append(node("p", `TeX SHA-256: ${state.resumeArtifact.tex_sha256}`));
    for (const instruction of state.resumeArtifact.compile.instructions) delivery.append(node("p", instruction));
    const download = node("button", "Download this draft .tex");
    download.type = "button";
    download.addEventListener("click", () => downloadResumeTex().catch(reportInterfaceFailure));
    delivery.append(download);
  }
  renderResumeRevision();
}

async function readJobInput() {
  const file = byId("resume-job-file").files?.[0];
  if (file) {
    if (!/\.(txt|md)$/i.test(file.name) || file.name.length > 120 || /[/\\]/.test(file.name)) {
      throw new Error("Choose a .txt or .md filename without a path.");
    }
    const bytes = await file.arrayBuffer();
    if (bytes.byteLength > RESUME_JOB_MAX_BYTES) throw new Error("JD exceeds 32 KiB.");
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    if (!text.trim()) throw new Error("JD is empty.");
    return { source: "upload", filename: file.name, text };
  }
  const text = byId("resume-job-paste").value;
  if (!text.trim() || new TextEncoder().encode(text).byteLength > RESUME_JOB_MAX_BYTES) {
    throw new Error("Paste a nonempty JD of at most 32 KiB, or select a text file.");
  }
  return { source: "paste", text };
}

async function createResumeSubmission() {
  if (!state.workspace || !state.resumeProfile || state.resumeSubmission || state.resumeSubmitting
      || state.resumePreparing) return;
  const generation = state.resumeGeneration;
  state.resumePreparing = true;
  renderResumeGenerationSetup();
  let submission;
  try {
    const projectIds = Array.from(byId("resume-job-projects").querySelectorAll("input:checked"))
      .map(item => item.value);
    if (!projectIds.length) throw new Error("Select at least one project.");
    if (!crypto.randomUUID) throw new Error("Secure request ID unavailable.");
    const job = await readJobInput();
    if (generation !== state.resumeGeneration) return;
    if (job.source === "upload") {
      byId("resume-job-file-preview").textContent = `${job.filename} · ${new TextEncoder().encode(job.text).byteLength} UTF-8 bytes`;
    }
    const calls = Number(byId("resume-job-model-calls").value);
    const tools = Number(byId("resume-job-tool-calls").value);
    const cost = byId("resume-job-cost").value;
    if (!Number.isInteger(calls) || calls < 1 || calls > 12 || !Number.isInteger(tools)
        || tools < 0 || tools > 8 || !/^\d+(?:\.\d{1,6})?$/.test(cost)
        || cost.replace(".", "").length > 12 || Number(cost) <= 0) {
      throw new Error("Enter valid positive model and cost limits and a tool limit from 0 to 8.");
    }
    const override = { page_target: Number(byId("resume-job-pages").value) };
    const advice = byId("resume-job-advice").value;
    const bullets = byId("resume-job-bullets").value;
    if (advice) override.writing_advice = advice;
    if (bullets !== "") override.max_bullets_per_project = Number(bullets);
    submission = {
      key: crypto.randomUUID(), actorId: state.me.user_id, workspaceId: state.workspace.workspace_id,
      conflict: false, accepted: null,
      body: {
        profile_version_id: state.resumeProfile.version_id,
        preference_version: state.resumeProfile.preference_version,
        project_ids: projectIds, job, override,
        budget: { max_model_calls: calls, max_tool_calls: tools, max_cost_cny: cost },
      },
    };
    state.resumeSubmission = submission;
  } finally {
    if (generation === state.resumeGeneration) {
      state.resumePreparing = false;
      renderResumeGenerationSetup();
    }
  }
  await sendResumeSubmission(submission);
}

async function sendResumeSubmission(submission) {
  if (!submission || submission !== state.resumeSubmission || submission.conflict
      || submission.accepted || state.resumeSubmitting || submission.actorId !== state.me?.user_id
      || submission.workspaceId !== state.workspace?.workspace_id) return;
  state.resumeSubmitting = true;
  byId("resume-job-retry").hidden = true;
  byId("resume-job-status").textContent = "Submitting the fixed request…";
  try {
    const receipt = await apiFetch(resumeBase(), {
      method: "POST", headers: { "Idempotency-Key": submission.key }, body: submission.body,
      signal: state.resumeRequestController.signal,
    });
    if (submission !== state.resumeSubmission) return;
    if (!receipt.session_id || !receipt.run_id) throw new Error("Invalid session receipt.");
    submission.accepted = receipt;
    byId("resume-job-new").hidden = false;
    byId("resume-job-status").textContent = "Accepted. Reading the saved job…";
    await openResumeSession(receipt.session_id);
    await loadResumeSessions();
  } catch (error) {
    if (submission !== state.resumeSubmission) return;
    submission.conflict = error.problem?.status === 409;
    byId("resume-job-retry").hidden = submission.conflict || Boolean(submission.accepted);
    byId("resume-job-new").hidden = !submission.conflict;
    byId("resume-job-status").textContent = submission.conflict
      ? "Request conflict. Your inputs remain; review before starting a new request."
      : submission.accepted ? "Accepted, but reading failed. Refresh the saved job."
      : "Response uncertain. Retry uses the same key and original JD.";
    throw error;
  } finally {
    if (submission === state.resumeSubmission) {
      state.resumeSubmitting = false;
      renderResumeGenerationSetup();
    }
  }
}

async function cancelResumeSession() {
  const sessionId = state.resumeSessionId;
  if (!sessionId || state.resumeTerminal) return;
  await apiFetch(`${resumeBase()}/${sessionId}/cancel`, {
    method: "POST", signal: state.resumeRequestController.signal,
  });
  await refreshResumeDetail(sessionId, state.resumeGeneration);
}

async function fetchResumeEventStream(url, signal, refreshed = false) {
  const headers = new Headers({ Accept: "text/event-stream", "Last-Event-ID": String(state.resumeLastEventId) });
  if (state.config.auth_mode === "supabase" && state.accessToken) headers.set("Authorization", `Bearer ${state.accessToken}`);
  const response = await fetch(url, { headers, signal });
  if (response.status === 401 && state.config.auth_mode === "supabase" && !refreshed
      && await refreshSession(signal)) return fetchResumeEventStream(url, signal, true);
  if (!response.ok) throw new PathfinderProblem(await problemFromResponse(response));
  return response;
}

async function consumeResumeEvents(response, sessionId, generation) {
  if (!response.body) throw new Error("Event stream body unavailable.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (generation === state.resumeGeneration && !state.resumeTerminal) {
      const chunk = await reader.read();
      buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
      let boundary = buffer.search(/\r?\n\r?\n/);
      while (boundary >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary).replace(/^\r?\n\r?\n/, "");
        if (frame && !frame.startsWith(":")) {
          let id = null, eventType = "", dataText = "";
          for (const line of frame.split(/\r?\n/)) {
            if (line.startsWith("id:")) id = Number(line.slice(3).trim());
            else if (line.startsWith("event:")) eventType = line.slice(6).trim();
            else if (line.startsWith("data:")) dataText += line.slice(5).trimStart();
          }
          if (!Number.isSafeInteger(id) || !dataText) throw new Error("Invalid event frame.");
          if (id > state.resumeLastEventId) {
            const data = JSON.parse(dataText);
            if (data.run_id !== state.resumeSession?.run_id || data.seq !== id) throw new Error("Event identity mismatch.");
            if (["run.status_changed", "run.completed", "run.failed", "run.cancelled"].includes(eventType)) {
              await refreshResumeDetail(sessionId, generation);
              if (generation !== state.resumeGeneration) return;
            }
            state.resumeLastEventId = id;
          }
        }
        boundary = buffer.search(/\r?\n\r?\n/);
      }
      if (chunk.done) return;
    }
  } finally {
    try { await reader.cancel(); } catch (_error) { /* Stream may already be closed. */ }
    reader.releaseLock();
  }
}

async function streamResumeEvents(sessionId, generation) {
  const controller = new AbortController();
  state.resumeStreamController = controller;
  const url = `/api/v1/workspaces/${state.workspace.workspace_id}/runs/${state.resumeSession.run_id}/events`;
  const delays = [500, 1000, 2000, 4000, 5000];
  let attempts = 0;
  try {
    while (!state.resumeTerminal && generation === state.resumeGeneration) {
      const cursor = state.resumeLastEventId;
      try {
        const response = await fetchResumeEventStream(url, controller.signal);
        await consumeResumeEvents(response, sessionId, generation);
        if (state.resumeTerminal || generation !== state.resumeGeneration) return;
      } catch (error) {
        if (controller.signal.aborted || generation !== state.resumeGeneration) return;
        if (error.problem?.status >= 400 && error.problem.status < 500
            && ![408, 429].includes(error.problem.status)) {
          showProblem(error.problem, "stream");
          return;
        }
      }
      if (state.resumeLastEventId > cursor) attempts = 0;
      if (attempts >= delays.length) {
        byId("resume-job-status").textContent = "Event connection paused. Refresh the job to resume.";
        return;
      }
      await waitForReconnect(delays[attempts], controller.signal);
      attempts += 1;
    }
  } catch (error) {
    if (error.name !== "AbortError") throw error;
  } finally {
    if (state.resumeStreamController === controller) state.resumeStreamController = null;
  }
}

async function downloadResumeTex() {
  const artifact = state.resumeArtifact;
  const version = state.resumeVersion;
  if (!artifact || !version || !state.workspace) return;
  const path = `/api/v2/workspaces/${state.workspace.workspace_id}/artifacts/${artifact.artifact_id}/download`;
  const request = async (refreshed = false) => {
    const headers = new Headers();
    if (state.config.auth_mode === "supabase" && state.accessToken) headers.set("Authorization", `Bearer ${state.accessToken}`);
    const response = await fetch(path, { headers, signal: state.resumeRequestController.signal });
    if (response.status === 401 && !refreshed && state.config.auth_mode === "supabase"
        && await refreshSession(state.resumeRequestController.signal)) return request(true);
    if (!response.ok) throw new PathfinderProblem(await problemFromResponse(response));
    if (response.headers.get("x-content-sha256") !== artifact.tex_sha256) throw new Error("TeX digest mismatch.");
    return response.blob();
  };
  const blob = await request();
  if (artifact !== state.resumeArtifact || version !== state.resumeVersion) return;
  const url = URL.createObjectURL(blob);
  const link = node("a");
  link.href = url;
  link.download = `resume-${version.version_id}.tex`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function initializeResumeGeneration() {
  state.resumeGeneration = 0;
  resetResumeGeneration();
  byId("resume-job-form").addEventListener("submit", event => {
    event.preventDefault();
    createResumeSubmission().catch(reportInterfaceFailure);
  });
  byId("resume-job-retry").addEventListener("click", () => {
    sendResumeSubmission(state.resumeSubmission).catch(reportInterfaceFailure);
  });
  byId("resume-job-new").addEventListener("click", () => {
    state.resumeSubmission = null;
    byId("resume-job-new").hidden = true;
    byId("resume-job-retry").hidden = true;
    renderResumeGenerationSetup();
  });
  byId("resume-job-history").addEventListener("change", event => {
    openResumeSession(event.target.value).catch(reportInterfaceFailure);
  });
  byId("resume-job-refresh").addEventListener("click", () => {
    loadResumeSessions().catch(reportInterfaceFailure);
  });
  initializeResumeRevision();
  byId("resume-job-file").addEventListener("change", () => {
    byId("resume-job-file-preview").textContent = byId("resume-job-file").files?.[0]?.name || "";
  });
}

function resumeRevisionItems(content) {
  const items = [];
  if (!content) return items;
  for (const project of content.projects || []) {
    items.push([project.id, `Project: ${project.title.text}`]);
    (project.bullet_ids || []).forEach((id, index) =>
      items.push([id, `Bullet: ${project.bullets[index]?.text || id}`]));
  }
  for (const item of content.education || []) items.push([item.id, `Education: ${item.institution.text}`]);
  for (const item of content.skills || []) items.push([item.id, `Skill: ${item.label}`]);
  return items;
}

function revisionSelect(id, entries) {
  const element = byId(id);
  const selected = element.value;
  clear(element);
  for (const [value, label] of entries) {
    const option = node("option", label);
    option.value = value;
    element.append(option);
  }
  if (entries.some(([value]) => value === selected)) element.value = selected;
}

function renderResumeRevision() {
  const detail = state.resumeSession;
  const version = state.resumeVersion;
  if (!detail) return;
  revisionSelect("resume-feedback-target", resumeRevisionItems(version?.content));
  revisionSelect("resume-feedback-fact-ref", (version?.facts || []).map(fact =>
    [fact.version_id, `${fact.claim} (${fact.source})`]));
  revisionSelect("resume-feedback-question", (state.resumeQuestions || []).map(question =>
    [question.id, question.text]));
  revisionSelect("resume-feedback-project", (detail.project_ids || []).map(id =>
    [id, state.materialProjects.find(project => project.id === id)?.name || id]));
  const diff = byId("resume-revision-diff");
  clear(diff);
  diff.append(node("h4", `Version ${version?.version || "pending"} · session revision ${detail.revision}`));
  if (version?.parent_version_id) diff.append(node("p", `Previous version: ${version.parent_version_id}`));
  for (const change of version?.diff || []) {
    diff.append(node("p", `${change.operation} · ${change.item_id}${change.field ? ` · ${change.field}` : ""}`));
  }
  for (const impact of version?.impact || []) diff.append(node("p", impact));
  for (const item of state.resumeFeedbackHistory || []) {
    if (item.run_id && item.run_status !== "completed") {
      diff.append(node("p", `Feedback ${item.kind}: ${item.run_status}`));
    }
    for (const question of item.questions || []) diff.append(node("p", question));
  }
  const facts = byId("resume-revision-facts");
  clear(facts);
  facts.append(node("h4", "User provided facts"));
  for (const fact of state.resumeUserFacts || []) {
    const card = node("article", null, "evidence");
    card.append(node("p", `${fact.claim} · ${fact.scope} · ${fact.review_status}`));
    card.append(node("pre", JSON.stringify(fact.conditions, null, 2)));
    if (fact.review_status === "pending") {
      for (const decision of ["confirm", "reject"]) {
        const button = node("button", decision === "confirm" ? "I verify this fact" : "Reject fact");
        button.type = "button";
        button.addEventListener("click", () => reviewResumeFact(fact, decision).catch(reportInterfaceFailure));
        card.append(button);
      }
    }
    facts.append(card);
  }
  const locks = byId("resume-revision-locks");
  clear(locks);
  locks.append(node("h4", "Content locks"));
  for (const [id, label] of resumeRevisionItems(version?.content)) {
    const locked = (detail.locked_item_ids || []).includes(id);
    const button = node("button", `${locked ? "Unlock" : "Lock"}: ${label}`);
    button.type = "button";
    button.addEventListener("click", () => changeResumeLock(id, !locked).catch(reportInterfaceFailure));
    locks.append(button);
  }
}

function resumeFeedbackBody() {
  const detail = state.resumeSession;
  if (!detail) throw new Error("Open a saved job first.");
  const base = { expected_session_revision: detail.revision, base_version_id: detail.current_version_id };
  const kind = byId("resume-feedback-kind").value;
  const text = byId("resume-feedback-text").value.trim();
  if (kind === "content") {
    if (!detail.current_version_id) throw new Error("A draft version is required for content edits.");
    const itemId = byId("resume-feedback-target").value;
    if (!itemId) throw new Error("Choose a target item.");
    const operation = byId("resume-feedback-operation").value;
    if (operation === "instruction") {
      if (!text) throw new Error("Describe the requested local edit.");
      return { ...base, kind, target_item_ids: [itemId], instruction: text };
    }
    const patch = { operation, item_id: itemId };
    if (operation === "replace_text") {
      if (!text || !byId("resume-feedback-fact-ref").value) throw new Error("Provide text and a confirmed fact reference.");
      patch.field = byId("resume-feedback-field").value;
      patch.text = text;
      patch.fact_version_ids = [byId("resume-feedback-fact-ref").value];
    } else if (operation === "move") patch.position = Number(byId("resume-feedback-position").value);
    return { ...base, kind, target_item_ids: [itemId], patches: [patch] };
  }
  if (kind === "answer") {
    if (!byId("resume-feedback-question").value || !text) throw new Error("Choose a question and enter an answer.");
    return { ...base, kind, question_id: byId("resume-feedback-question").value, answer: text };
  }
  if (kind === "fact") {
    if (!text) throw new Error("Enter the fact to verify.");
    return { ...base, kind, fact: {
      project_id: byId("resume-feedback-project").value,
      scope: byId("resume-feedback-scope").value,
      kind: byId("resume-feedback-fact-kind").value,
      claim: text,
      environment: byId("resume-feedback-environment").value || null,
      fact_scope: byId("resume-feedback-metric-scope").value || null,
      metric_basis: byId("resume-feedback-metric-basis").value || null,
    } };
  }
  const scope = byId("resume-feedback-pref-scope").value;
  const page_target = Number(byId("resume-feedback-pages").value);
  if (scope === "global") {
    if (!state.resumeProfile?.preferences) throw new Error("Read the current profile first.");
    return { ...base, kind, scope, preferences: { ...state.resumeProfile.preferences, page_target },
      expected_global_preference_version: state.resumeProfile.preference_version };
  }
  return { ...base, kind, scope, preferences: { page_target } };
}

async function sendResumeFeedback(submission) {
  if (!submission || submission !== state.resumeFeedbackSubmission || submission.conflict
      || submission.actorId !== state.me?.user_id || submission.workspaceId !== state.workspace?.workspace_id) return;
  byId("resume-feedback-retry").hidden = true;
  byId("resume-feedback-status").textContent = "Saving feedback…";
  try {
    await apiFetch(`${resumeBase()}/${submission.sessionId}/feedback`, {
      method: "POST", headers: { "Idempotency-Key": submission.key }, body: submission.body,
      signal: state.resumeRequestController.signal,
    });
    if (submission !== state.resumeFeedbackSubmission) return;
    state.resumeFeedbackSubmission = null;
    byId("resume-feedback-status").textContent = "Feedback saved. Reading its progress…";
    await openResumeSession(submission.sessionId);
  } catch (error) {
    if (submission !== state.resumeFeedbackSubmission) return;
    submission.conflict = error.problem?.status === 409;
    byId("resume-feedback-retry").hidden = submission.conflict;
    byId("resume-feedback-status").textContent = submission.conflict
      ? "Session changed. Your inputs remain. Refresh the job, review, then submit a new request."
      : "Response uncertain. Retry uses the same key and saved feedback.";
    throw error;
  }
}

async function reviewResumeFact(fact, decision) {
  const detail = state.resumeSession;
  if (!detail) return;
  await apiFetch(`${resumeBase()}/${detail.session_id}/user-facts/${fact.fact_version_id}/reviews`, {
    method: "POST", headers: { "Idempotency-Key": crypto.randomUUID() },
    body: { expected_session_revision: detail.revision, base_version_id: detail.current_version_id,
      fact_version_id: fact.fact_version_id, decision, attested: decision === "confirm" },
    signal: state.resumeRequestController.signal,
  });
  await openResumeSession(detail.session_id);
}

async function changeResumeLock(itemId, locked) {
  const detail = state.resumeSession;
  if (!detail?.current_version_id) return;
  await apiFetch(`${resumeBase()}/${detail.session_id}/locks`, {
    method: "POST", headers: { "Idempotency-Key": crypto.randomUUID() },
    body: { expected_session_revision: detail.revision, base_version_id: detail.current_version_id,
      item_id: itemId, locked }, signal: state.resumeRequestController.signal,
  });
  await openResumeSession(detail.session_id);
}

function initializeResumeRevision() {
  byId("resume-feedback-form").addEventListener("submit", event => {
    event.preventDefault();
    try {
      const body = resumeFeedbackBody();
      const submission = { key: crypto.randomUUID(), body, sessionId: state.resumeSession.session_id,
        actorId: state.me.user_id, workspaceId: state.workspace.workspace_id, conflict: false };
      state.resumeFeedbackSubmission = submission;
      sendResumeFeedback(submission).catch(reportInterfaceFailure);
    } catch (error) {
      byId("resume-feedback-status").textContent = error.message;
    }
  });
  byId("resume-feedback-retry").addEventListener("click", () =>
    sendResumeFeedback(state.resumeFeedbackSubmission).catch(reportInterfaceFailure));
}
