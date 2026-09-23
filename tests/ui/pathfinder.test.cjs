"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const historicalSource = readFileSync(join(__dirname, "../fixtures/legacy_ui/pathfinder.js"), "utf8");
const resumeSource = readFileSync(join(__dirname, "../../src/app/api/static/resume_generation_ui.js"), "utf8");
const source = readFileSync(join(__dirname, "../../src/app/api/static/pathfinder.js"), "utf8");

const resumeDetail = (status = "completed", result = null) => ({
  session_id: "session-a", run_id: "run-a", run_status: status, error_category: null,
  result, current_version_id: null, revision: 0, profile_version_id: "profile-version-a",
  preference_version: 1, project_ids: ["project-a"], override: { page_target: 1 },
  budget: { max_model_calls: 6, max_tool_calls: 2, max_cost_cny: "2" },
  job: { source: "paste", filename: null, text: "synthetic job", sha256: "a".repeat(64) },
  requirements: [],
});

test("first draft retry retains key, JD and budget after an uncertain response", async t => {
  const h = await loadUi(t, { randomUUID: () => "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa" });
  h.state.resumeProfile = { version_id: "profile-version-a", version: 1,
    preference_version: 1, preferences: { page_target: 1 } };
  h.state.materialProjects = [{ id: "project-a", name: "Synthetic" }];
  h.call("renderResumeGenerationSetup");
  h.element("resume-job-projects").querySelector("input").checked = true;
  h.element("resume-job-paste").value = "original JD";
  h.element("resume-job-pages").value = "2";
  let posts = 0;
  h.route((url, options) => {
    if (options.method === "POST") {
      posts += 1;
      if (posts === 1) throw new Error("response lost");
      return json({ session_id: "session-a", run_id: "run-a" }, 202);
    }
    if (url.endsWith("/resume-sessions")) return json([{ session_id: "session-a", job_label: "synthetic job",
      run_status: "completed", created_at: "2026-09-23T00:00:00Z" }]);
    if (url.endsWith("/session-a")) return json(resumeDetail("completed",
      { outcome: "needs_input", questions: ["Need facts"] }));
    throw new Error(`Unexpected fetch ${url}`);
  });
  await assert.rejects(h.call("createResumeSubmission"), /response lost/);
  h.element("resume-job-paste").value = "changed JD";
  h.element("resume-job-cost").value = "5";
  await h.call("sendResumeSubmission", h.state.resumeSubmission);
  const requests = h.calls.filter(call => call.method === "POST");
  assert.equal(requests.length, 2);
  assert.equal(requests[0].body, requests[1].body);
  assert.equal(requests[0].headers.get("Idempotency-Key"), requests[1].headers.get("Idempotency-Key"));
  const body = JSON.parse(requests[0].body);
  assert.equal(body.job.text, "original JD");
  assert.deepEqual(body.project_ids, ["project-a"]);
  assert.equal(body.override.page_target, 2);
  assert.equal(body.budget.max_cost_cny, "2");
  assert.match(h.element("resume-job-draft").textContent, /No draft yet/);
});

test("JD upload rejects invalid UTF-8 and oversized content before POST", async t => {
  const h = await loadUi(t);
  h.element("resume-job-file").files = [{ name: "job.txt", arrayBuffer: async () => Uint8Array.of(255).buffer }];
  await assert.rejects(h.call("readJobInput"));
  h.element("resume-job-file").files = [{ name: "job.md", arrayBuffer: async () => new Uint8Array(32769).buffer }];
  await assert.rejects(h.call("readJobInput"), /32 KiB/);
  h.element("resume-job-file").files = [{ name: "job.md", arrayBuffer: async () => new TextEncoder().encode("Synthetic JD").buffer }];
  const job = await h.call("readJobInput");
  assert.equal(job.source, "upload");
  assert.equal(job.text, "Synthetic JD");
  assert.equal(h.calls.length, 0);
});

test("409 keeps current JD and blocks blind first draft retry", async t => {
  const h = await loadUi(t, { randomUUID: () => "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa" });
  h.state.resumeProfile = { version_id: "profile-version-a", version: 1,
    preference_version: 1, preferences: { page_target: 1 } };
  h.state.materialProjects = [{ id: "project-a", name: "Synthetic" }];
  h.call("renderResumeGenerationSetup");
  h.element("resume-job-projects").querySelector("input").checked = true;
  h.element("resume-job-paste").value = "Keep this JD";
  h.route(() => failure(409));
  await assert.rejects(h.call("createResumeSubmission"));
  await h.call("sendResumeSubmission", h.state.resumeSubmission);
  assert.equal(h.calls.length, 1);
  assert.equal(h.element("resume-job-paste").value, "Keep this JD");
  assert.equal(h.element("resume-job-retry").hidden, true);
  assert.equal(h.element("resume-job-new").hidden, false);
});

test("saved session is recovered by list and URL without a new POST", async t => {
  const h = await loadUi(t);
  h.call("rememberResumeSession", "session-a");
  h.route(url => url.endsWith("/resume-sessions")
    ? json([{ session_id: "session-a", job_label: "synthetic job", run_status: "completed",
      created_at: "2026-09-23T00:00:00Z" }]) : json(resumeDetail()));
  await h.call("loadResumeSessions");
  assert.equal(h.state.resumeSessionId, "session-a");
  assert.equal(h.state.resumeSession.run_status, "completed");
  assert.ok(h.calls.every(call => (call.method || "GET") === "GET"));
});

test("coverage labels keep unchecked material separate from confirmed gap and render text safely", async t => {
  const h = await loadUi(t);
  const detail = resumeDetail();
  detail.requirements = [{ id: "req-a", kind: "inferred", quote: "<img src=x>",
    start: 0, end: 11, inference_basis: "untrusted JD" }];
  h.state.resumeSession = detail;
  h.state.resumeVersion = { content: { display_name: "Synthetic", education: [], projects: [], skills: [] },
    validation: {}, coverage: [{ requirement_id: "req-a", support: "no_support_found",
      verification: "unchecked", reason: "Nothing selected", fact_version_ids: [], item_ids: [] }], facts: [] };
  h.call("renderResumeSession");
  const panel = h.element("resume-job-coverage");
  assert.match(panel.textContent, /Not checked/);
  assert.doesNotMatch(panel.textContent, /Confirmed ability gap/);
  assert.match(panel.textContent, /<img src=x>/);
  assert.equal(panel.querySelector("img"), null);
});

test("SSE detail failure keeps cursor at zero until replay succeeds", async t => {
  const h = await loadUi(t);
  h.state.resumeSessionId = "session-a";
  h.state.resumeSession = resumeDetail("running");
  let reads = 0;
  h.route(url => {
    if (url.endsWith("/events")) return stream([frame(1, "run.completed")]);
    reads += 1;
    return reads === 1 ? failure(503) : json(resumeDetail("completed"));
  });
  const work = h.call("streamResumeEvents", "session-a", h.state.resumeGeneration);
  await until(() => h.clock.has(500));
  assert.equal(h.state.resumeLastEventId, 0);
  h.clock.fire(500);
  await work;
  assert.equal(h.state.resumeLastEventId, 1);
  assert.deepEqual(h.calls.filter(call => call.url.endsWith("/events"))
    .map(call => call.headers.get("Last-Event-ID")), ["0", "0"]);
});

test("TeX download uses authenticated bytes and fixed artifact digest", async t => {
  const h = await loadUi(t);
  h.state.config = { auth_mode: "supabase" };
  h.state.accessToken = "synthetic-token";
  h.state.resumeVersion = { version_id: "version-a" };
  h.state.resumeArtifact = { artifact_id: "artifact-a", tex_sha256: "a".repeat(64) };
  h.route((_url, options) => {
    assert.equal(options.headers.get("Authorization"), "Bearer synthetic-token");
    return new Response("synthetic tex", { headers: { "X-Content-SHA256": "a".repeat(64) } });
  });
  await h.call("downloadResumeTex");
  assert.equal(h.created.find(item => item.tagName === "a").download, "resume-version-a.tex");
  assert.equal(h.calls.length, 1);
});

test("resume preview and import retry preserve the exact source and key", async t => {
  const h = await loadUi(t, { randomUUID: () => "33333333-3333-4333-8333-333333333333" });
  h.element("resume-profile-file").files = [{ name: "synthetic.tex", text: async () => "PRIVATE-SYNTHETIC-CANARY" }];
  let attempts = 0;
  h.route((url, options) => {
    if (url.endsWith("/import-preview")) return json({
      template_commit: "synthetic", source_sha256: "a".repeat(64), complete: true,
      content: null, claims: [], issues: [],
    });
    if (url.endsWith("/imports")) {
      attempts += 1;
      if (attempts === 1) throw new Error("response lost");
      return json({ command_id: "command-a", resource_id: "profile-a", status: "completed", replayed: true }, 201);
    }
    if (url.endsWith("/profiles/me")) return failure(404);
    throw new Error(`Unexpected fetch ${url}`);
  });
  await h.call("previewResumeSource");
  assert.equal(h.element("resume-profile-import").disabled, false);
  await assert.rejects(h.call("queueResumeCommand", "/api/v2/workspaces/workspace-a/profiles/imports",
    { source_tex: h.state.resumeSourceTex }), /response lost/);
  h.element("resume-profile-file").files = [{ name: "changed.tex", text: async () => "CHANGED" }];
  await h.call("sendResumeCommand", h.state.resumePending);
  const requests = h.calls.filter(item => item.url.endsWith("/imports"));
  assert.equal(requests.length, 2);
  assert.equal(requests[0].headers.get("Idempotency-Key"), requests[1].headers.get("Idempotency-Key"));
  assert.equal(requests[0].body, requests[1].body);
  assert.match(requests[1].body, /PRIVATE-SYNTHETIC-CANARY/);
});

test("resume version conflict retains preference input and blocks blind retry", async t => {
  const h = await loadUi(t, { randomUUID: () => "44444444-4444-4444-8444-444444444444" });
  h.state.resumeProfile = { profile_id: "profile-a", preference_version: 1 };
  h.element("resume-pref-terms").value = "explicit synthetic term";
  h.element("resume-pref-order").value = "education,projects,skills";
  h.element("resume-pref-pages").value = "1";
  h.route(() => failure(409));
  await assert.rejects(h.call("saveResumePreferences"));
  assert.equal(h.element("resume-pref-terms").value, "explicit synthetic term");
  assert.equal(h.state.resumePending.conflict, true);
  assert.equal(h.element("resume-profile-retry").hidden, true);
});

test("material import retry keeps the original key and source selection", async t => {
  const h = await loadUi(t, { randomUUID: () => "11111111-1111-4111-8111-111111111111" });
  h.state.materialProjectId = "project-a";
  h.state.materialSources = [{ id: "source-a", alias: "synthetic", kind: "file" }];
  h.element("material-source-source-a").checked = true;
  let submissions = 0;
  h.route((url, options) => {
    if (url.endsWith("/imports") && options.method === "POST") {
      submissions += 1;
      if (submissions === 1) throw new Error("response lost");
      return json({ command_id: "command-a", run_id: "run-a", import_id: "import-a", status: "queued", replayed: true }, 202);
    }
    if (url.endsWith("/imports/import-a")) {
      return json({ id: "import-a", project_id: "project-a", run_id: "run-a", status: "completed", error_category: null, snapshots: [] });
    }
    if (url.endsWith("/projects/project-a/facts")) {
      return json({ fact_set_id: null, import_id: null, complete: false, issues: [], facts: [] });
    }
    throw new Error(`Unexpected fetch ${url}`);
  });
  await assert.rejects(h.call("submitMaterialImport"), /response lost/);
  assert.equal(h.element("material-retry-import").hidden, false);
  await h.call("sendMaterialImport", h.state.materialSubmission);
  const requests = h.calls.filter(item => item.url.endsWith("/imports") && item.method === "POST");
  assert.equal(requests.length, 2);
  assert.equal(requests[0].headers.get("Idempotency-Key"), requests[1].headers.get("Idempotency-Key"));
  assert.equal(requests[0].body, requests[1].body);
  assert.equal(h.element("material-import-status").textContent, "Import completed.");
});

test("empty material aliases disable import setup", async t => {
  const h = await loadUi(t);
  h.state.materialAliases = [];
  h.state.materialProjects = [];
  h.state.materialSources = [];
  h.call("renderMaterials");
  assert.equal(h.element("material-add-source").disabled, true);
  assert.equal(h.element("material-import").disabled, true);
  assert.match(h.element("materials-availability").textContent, /No material aliases/);
});

test("fact command retry keeps the same key and source text after response loss", async t => {
  const h = await loadUi(t, { randomUUID: () => "22222222-2222-4222-8222-222222222222" });
  h.state.materialProjectId = "project-a";
  h.state.materialFacts = { fact_set_id: "set-a", import_id: "import-a", complete: false, issues: [], facts: [] };
  h.element("material-fact-claim").value = "I prepared synthetic documentation";
  h.element("material-fact-kind").value = "personal_statement";
  let attempts = 0;
  h.route((url, options) => {
    if (url.endsWith("/projects/project-a/facts") && options.method === "POST") {
      attempts += 1;
      if (attempts === 1) throw new Error("response lost");
      return json({ command_id: "command-a", resource_id: "fact-a", status: "completed", replayed: true }, 201);
    }
    if (url.endsWith("/projects/project-a/facts")) {
      return json({ fact_set_id: "set-a", import_id: "import-a", complete: false, issues: [], facts: [] });
    }
    throw new Error(`Unexpected fetch ${url}`);
  });
  await assert.rejects(h.call("addMaterialFact"), /response lost/);
  assert.equal(h.element("material-fact-claim").value, "I prepared synthetic documentation");
  assert.equal(h.element("material-fact-retry").hidden, false);
  await h.call("sendMaterialFactCommand", h.state.materialFactPending);
  const requests = h.calls.filter(item => item.method === "POST" && item.url.endsWith("/projects/project-a/facts"));
  assert.equal(requests.length, 2);
  assert.equal(requests[0].headers.get("Idempotency-Key"), requests[1].headers.get("Idempotency-Key"));
  assert.equal(requests[0].body, requests[1].body);
});

test("fact search sends the selected project ids only", async t => {
  const h = await loadUi(t);
  h.state.materialProjects = [{ id: "project-a", name: "A" }, { id: "project-b", name: "B" }];
  h.state.materialProjectId = "project-a";
  h.call("renderMaterials");
  h.element("material-fact-search-query").value = "synthetic";
  h.route((url) => {
    assert.match(url, /project_ids=project-a/);
    assert.match(url, /project_ids=project-b/);
    return json([]);
  });
  await h.call("searchMaterialFacts");
  assert.match(h.element("material-search-results").textContent, /No confirmed facts/);
});

class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.listeners = new Map();
    this.hidden = true;
    this.value = "";
    this.text = "";
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(""); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.text = ""; this.children = children; }
  querySelector(tag) {
    for (const child of this.children) {
      if (child.tagName === tag) return child;
      const nested = child.querySelector(tag);
      if (nested) return nested;
    }
    return null;
  }
  querySelectorAll(selector) {
    if (selector !== "input:checked") return [];
    const found = [];
    for (const child of this.children) {
      if (child.tagName === "input" && child.checked) found.push(child);
      found.push(...child.querySelectorAll(selector));
    }
    return found;
  }
  addEventListener(type, callback) { this.listeners.set(type, callback); }
  click() { if (this.listeners.has("click")) this.listeners.get("click")(); else assert.equal(this.tagName, "a"); }
  reportValidity() { return true; }
  reset() {}
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

async function until(predicate) {
  for (let i = 0; i < 40; i += 1) {
    if (predicate()) return;
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.fail("Expected asynchronous boundary was not reached");
}

function fakeClock() {
  const pending = new Map();
  const fired = [];
  let nextId = 0;
  return {
    pending, fired,
    setTimeout(callback, delay) { const id = ++nextId; pending.set(id, { callback, delay }); return id; },
    clearTimeout(id) { pending.delete(id); },
    has(delay) { return [...pending.values()].some(timer => timer.delay === delay); },
    fire(delay) {
      const entry = [...pending].find(([, timer]) => timer.delay === delay);
      assert.ok(entry, `Missing ${delay}ms timer`);
      pending.delete(entry[0]);
      fired.push(delay);
      entry[1].callback();
    },
  };
}

const json = (value, status = 200) => new Response(JSON.stringify(value), {
  status,
  headers: { "Content-Type": status >= 400 ? "application/problem+json" : "application/json" },
});
const failure = status => json({ status, title: "Synthetic read failure", detail: "Try again" }, status);
const run = (id = "run-a", status = "running") => ({
  run_id: id, status, mode: "application", usage: { chat: {}, embedding: {} },
});
const action = (id = "action-a") => ({
  action_intent_id: id, status: "proposed", action_key: "submit_application", action_revision: 1,
  approval_request: { status: "pending", version: 1, expires_at: "2099-01-01T00:00:00Z" },
  args_snapshot: { text: "synthetic" }, target_snapshot: {},
});
const frame = (id, type, payload = {}) => `id: ${id}\nevent: ${type}\ndata: ${JSON.stringify({
  run_id: "run-a", seq: id, payload,
})}\n\n`;

function stream(frames, metrics = { cancelled: 0, released: 0 }) {
  const chunks = frames.map(value => new TextEncoder().encode(value));
  let offset = 0;
  return {
    ok: true, status: 200, headers: new Headers(),
    body: {
      getReader() {
        return {
          async read() {
            return offset < chunks.length
              ? { value: chunks[offset++], done: false } : { done: true };
          },
          async cancel() { metrics.cancelled += 1; },
          releaseLock() { metrics.released += 1; },
        };
      },
    },
  };
}

async function loadUi(t, { randomUUID, confirm = () => true, historical = false } = {}) {
  const elements = new Map();
  const created = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  element("run-mode").value = "research";
  const clock = fakeClock();
  const calls = [];
  let route = () => { throw new Error("Unexpected fetch"); };
  const context = vm.createContext({
    document: { getElementById: element, createElement: tag => {
      const item = new Element(tag); created.push(item); return item;
    } },
    window: { setTimeout: clock.setTimeout, clearTimeout: clock.clearTimeout, confirm,
      location: { search: "" }, history: { replaceState(_state, _title, path) {
        this.path = path;
        context.window.location.search = path.split("?")[1] || "";
      } } },
    crypto: { randomUUID },
    Headers, AbortController, DOMException, TextDecoder, TextEncoder, Uint8Array, URL, URLSearchParams, Blob,
    fetch: async (url, options = {}) => {
      if (url === "/api/v1/ui-config") return json({ auth_mode: "supabase" });
      calls.push({ url, ...options });
      if (options.signal?.aborted) throw new DOMException("Aborted", "AbortError");
      let abort;
      const interrupted = new Promise((resolve, reject) => {
        abort = () => reject(new DOMException("Aborted", "AbortError"));
        options.signal?.addEventListener("abort", abort, { once: true });
      });
      try {
        return await Promise.race([Promise.resolve().then(() => route(url, options)), interrupted]);
      } finally {
        options.signal?.removeEventListener("abort", abort);
      }
    },
  });
  if (!historical) vm.runInContext(resumeSource, context, { filename: "resume_generation_ui.js" });
  vm.runInContext(historical ? historicalSource : source, context, { filename: "pathfinder.js" });
  const state = vm.runInContext("state", context);
  await until(() => state.config !== null);
  Object.assign(state, {
    config: { auth_mode: "fake" }, me: { user_id: "actor-a" },
    workspace: { workspace_id: "workspace-a", role: "admin" }, run: run(),
  });
  const call = (name, ...args) => {
    context.__arguments = args;
    return vm.runInContext(`${name}(...__arguments)`, context);
  };
  t.after(async () => { call("resetProjection"); await new Promise(resolve => setImmediate(resolve)); });
  return { state, clock, calls, created, element, call, route: handler => { route = handler; } };
}

for (const type of ["action.proposed", "run.completed", "run.failed", "run.cancelled"]) {
  test(`${type}: failed detail is unacknowledged and replay restores projection`, { timeout: 5000 }, async t => {
    const h = await loadUi(t);
    const proposal = type === "action.proposed";
    const first = frame(1, type, { action_intent_id: "action-a" });
    const terminal = proposal ? frame(2, "run.completed") : "";
    const streams = [];
    let reads = 0;
    h.route((url) => {
      if (url.endsWith("/events")) {
        const metrics = { cancelled: 0, released: 0 };
        streams.push(metrics);
        return stream([first, first, terminal].filter(Boolean), metrics);
      }
      reads += 1;
      if (reads === 1) return failure(503);
      return json(url.includes("action-intents") ? action() : run("run-a", proposal ? "completed" : type.slice(4)));
    });
    const work = h.call("streamEvents", "/api/v1/workspaces/workspace-a/runs/run-a/events");
    await until(() => h.clock.has(500));
    assert.equal(h.state.lastEventId, 0);
    assert.equal(h.state.terminal, false);
    assert.equal(h.state.run.status, "running");
    assert.equal(h.element("timeline-panel").querySelector("ol"), null);
    assert.deepEqual(streams[0], { cancelled: 1, released: 1 });
    h.clock.fire(500);
    await work;
    assert.equal(h.state.terminal, true);
    assert.equal(h.state.lastEventId, proposal ? 2 : 1);
    assert.equal(h.element("timeline-panel").querySelector("ol").children.length, proposal ? 2 : 1);
    if (proposal) {
      assert.equal(h.state.action.action_intent_id, "action-a");
      assert.equal(h.element("action-panel").hidden, false);
      assert.equal(h.element("action-panel").querySelector("button"), null);
    }
    assert.ok(h.calls.every(call => (call.method || "GET") === "GET"));
    assert.deepEqual(h.calls.filter(call => call.url.endsWith("/events")).map(call => call.headers.get("Last-Event-ID")), ["0", "0"]);
    assert.ok(streams.every(metrics => metrics.cancelled === 1 && metrics.released === 1));
    assert.equal(h.clock.pending.size, 0);
  });
}

test("five unsuccessful reconnects pause and manual recovery only reads the retained run", { timeout: 5000 }, async t => {
  const h = await loadUi(t);
  h.state.lastEventId = 7;
  h.route(() => failure(503));
  const work = h.call("streamEvents", "/api/v1/workspaces/workspace-a/runs/run-a/events");
  for (const delay of [500, 1000, 2000, 4000, 5000]) {
    await until(() => h.clock.has(delay));
    h.clock.fire(delay);
  }
  await work;
  assert.equal(h.calls.length, 6);
  assert.equal(h.state.lastEventId, 7);
  assert.equal(h.element("retry-events").hidden, false);
  assert.match(h.element("problem-panel").textContent, /Event recovery paused/);
  h.route(url => url.endsWith("/events") ? stream([frame(8, "run.completed")]) : json(run("run-a", "completed")));
  h.element("retry-events").click();
  await until(() => h.state.terminal);
  assert.equal(h.state.lastEventId, 8);
  assert.equal(h.element("retry-events").hidden, true);
  assert.ok(h.calls.every(call => (call.method || "GET") === "GET"));
  assert.equal(h.calls.filter(call => call.url.endsWith("/events")).at(-1).headers.get("Last-Event-ID"), "7");
});

test("HTTP 200 without event progress does not reset retry budget", { timeout: 5000 }, async t => {
  const h = await loadUi(t);
  h.route(() => stream([]));
  const work = h.call("streamEvents", "/events");
  for (const delay of [500, 1000, 2000, 4000, 5000]) {
    await until(() => h.clock.has(delay));
    h.clock.fire(delay);
  }
  await work;
  assert.equal(h.calls.length, 6);
  assert.equal(h.element("retry-events").hidden, false);
});

test("only acknowledged progress resets the reconnect delay", { timeout: 5000 }, async t => {
  const h = await loadUi(t);
  let connects = 0;
  h.route(url => {
    if (!url.endsWith("events")) return json(run("run-a", "completed"));
    connects += 1;
    if (connects < 3) return failure(503);
    return stream([frame(connects === 3 ? 1 : 2, connects === 3 ? "job.queued" : "run.completed")]);
  });
  const work = h.call("streamEvents", "/events");
  for (const delay of [500, 1000, 500]) {
    await until(() => h.clock.has(delay));
    h.clock.fire(delay);
  }
  await work;
  assert.equal(h.state.lastEventId, 2);
  assert.deepEqual(h.clock.fired, [500, 1000, 500]);
});

for (const status of [400, 401, 403, 404, 409, 422]) {
  test(`definitive HTTP ${status} stops without automatic retry`, { timeout: 5000 }, async t => {
    const h = await loadUi(t);
    h.route(() => failure(status));
    await h.call("streamEvents", "/events");
    assert.equal(h.calls.length, 1);
    assert.equal(h.clock.pending.size, 0);
    assert.equal(h.state.terminal, false);
    assert.equal(h.element("retry-events").hidden, false);
  });
}

for (const status of [408, 429, 500, 503]) {
  test(`temporary HTTP ${status} retries reads`, { timeout: 5000 }, async t => {
    const h = await loadUi(t);
    let connections = 0;
    h.route(url => {
      if (url.endsWith("/events")) return ++connections === 1 ? failure(status) : stream([frame(1, "run.completed")]);
      return json(run("run-a", "completed"));
    });
    const work = h.call("streamEvents", "/events");
    await until(() => h.clock.has(500));
    h.clock.fire(500);
    await work;
    assert.equal(connections, 2);
    assert.equal(h.state.terminal, true);
  });
}

test("detail timeout aborts GET, releases reader and replays without advancing cursor", { timeout: 5000 }, async t => {
  const h = await loadUi(t);
  let reads = 0;
  const metrics = { cancelled: 0, released: 0 };
  h.route(url => {
    if (url.endsWith("/events")) return stream([frame(1, "run.completed")], metrics);
    reads += 1;
    return reads === 1 ? new Promise(() => {}) : json(run("run-a", "completed"));
  });
  const work = h.call("streamEvents", "/events");
  await until(() => h.clock.has(30000) && reads === 1);
  const detailSignal = h.calls.at(-1).signal;
  h.clock.fire(30000);
  await until(() => h.clock.has(500));
  assert.equal(detailSignal.aborted, true);
  assert.equal(h.state.lastEventId, 0);
  assert.equal(h.state.terminal, false);
  assert.deepEqual(metrics, { cancelled: 1, released: 1 });
  h.clock.fire(500);
  await work;
  assert.equal(h.state.terminal, true);
  assert.equal(h.clock.pending.size, 0);
});

test("switching projection during detail read cancels and cannot acknowledge old terminal", { timeout: 5000 }, async t => {
  const h = await loadUi(t);
  const metrics = { cancelled: 0, released: 0 };
  h.route(url => url.endsWith("/events") ? stream([frame(1, "run.completed")], metrics) : new Promise(() => {}));
  const work = h.call("streamEvents", "/events");
  await until(() => h.clock.has(30000));
  h.call("resetProjection");
  h.state.run = run("run-b");
  h.call("showProblem", { title: "Current task message" }, "readRun");
  await work;
  assert.equal(h.state.run.run_id, "run-b");
  assert.equal(h.state.lastEventId, 0);
  assert.equal(h.state.terminal, false);
  assert.match(h.element("problem-panel").textContent, /Current task message/);
  assert.deepEqual(metrics, { cancelled: 1, released: 1 });
  assert.equal(h.clock.pending.size, 0);
});

test("aborting a reconnect cancels its timer and hides stale recovery", { timeout: 5000 }, async t => {
  const h = await loadUi(t);
  h.route(() => failure(503));
  const work = h.call("streamEvents", "/events");
  await until(() => h.clock.has(500));
  h.call("resetProjection");
  await work;
  assert.equal(h.calls.length, 1);
  assert.equal(h.clock.pending.size, 0);
  assert.equal(h.element("retry-events").hidden, true);
});

for (const stage of ["post", "get"]) {
  for (const outcome of ["success", "conflict", "read-failure"]) {
    for (const change of ["run", "workspace", "logout"]) {
      test(`late approval ${stage}/${outcome} cannot update after ${change}`, { timeout: 5000 }, async t => {
        const h = await loadUi(t, { historical: true });
        h.state.action = action();
        const delayed = deferred();
        h.route((url, options) => {
          if (options.method === "POST") {
            if (stage === "post") return delayed.promise;
            return outcome === "conflict" ? failure(409) : json({});
          }
          return delayed.promise;
        });
        const work = h.call("submitDecision", "approve", " synthetic ");
        await until(() => h.calls.length === (stage === "post" ? 1 : 2));
        if (change === "logout") h.call("logout", true);
        else {
          h.call("resetProjection");
          if (change === "workspace") {
            h.call("invalidateSubmission");
            h.state.workspace = { workspace_id: "workspace-b", role: "admin" };
          }
          h.state.run = run("run-b");
          h.call("renderAction", action("action-b"));
        }
        h.call("showProblem", { title: "Current task message" }, "readRun");
        const panelBefore = h.element("action-panel").textContent;
        if (outcome === "read-failure") delayed.reject(new TypeError("Old network failure"));
        else delayed.resolve(outcome === "conflict" && stage === "post" ? failure(409) : json(action()));
        await work;
        assert.equal(h.state.action?.action_intent_id, change === "logout" ? undefined : "action-b");
        assert.equal(h.element("action-panel").textContent, panelBefore);
        assert.match(h.element("problem-panel").textContent, /Current task message/);
        assert.equal(h.calls.filter(call => call.method === "POST").length, 1);
        assert.equal(h.calls.length, stage === "post" ? 1 : 2);
        assert.deepEqual(JSON.parse(h.calls[0].body), { decision: "approve", expected_version: 1, reason: "synthetic" });
      });
    }
  }
}

for (const status of [200, 409]) {
  test(`current approval ${status} refreshes the same action once`, { timeout: 5000 }, async t => {
    const h = await loadUi(t, { historical: true });
    h.state.action = action();
    h.route((url, options) => options.method === "POST"
      ? (status === 409 ? failure(409) : json({}))
      : json({ ...action(), status: "authorized" }));
    await h.call("submitDecision", "approve", "");
    assert.equal(h.state.action.status, "authorized");
    assert.equal(h.calls.length, 2);
    assert.equal(h.element("problem-panel").hidden, status === 200);
  });
}

for (const kind of ["run", "action"]) {
  test(`manual ${kind} read still reports failure without starting recovery`, { timeout: 5000 }, async t => {
    const h = await loadUi(t, { historical: true });
    h.route(() => failure(503));
    if (kind === "run") await h.call("fetchRun");
    else await h.call("fetchAction", "action-a");
    assert.equal(h.calls.length, 1);
    assert.equal(h.element("problem-panel").hidden, false);
    assert.equal(h.state.lastEventId, 0);
    assert.equal(h.clock.pending.size, 0);
  });
}

test("revisiting the same resource cannot revive an old approval generation", { timeout: 5000 }, async t => {
  const h = await loadUi(t, { historical: true });
  h.state.action = action();
  const old = deferred();
  h.route((url, options) => options.method === "POST" ? json({}) : old.promise);
  const work = h.call("submitDecision", "approve", "");
  await until(() => h.calls.length === 2);
  h.call("resetProjection");
  h.state.run = run();
  h.call("renderAction", action());
  h.call("showProblem", { title: "New visit" }, "readRun");
  old.resolve(json({ ...action(), status: "authorized" }));
  await work;
  assert.equal(h.state.action.status, "proposed");
  assert.match(h.element("problem-panel").textContent, /New visit/);
});

// Historical E3.6: execute the frozen submission lifecycle with controlled HTTP outcomes.
const requestKey = "12345678-1234-4234-9234-123456789abc";
const nextRequestKey = "22345678-1234-4234-9234-123456789abc";
const acceptedRunId = "32345678-1234-4234-9234-123456789abc";
const receipt = () => ({
  run_id: acceptedRunId, status: "queued",
  events_url: `/api/v1/workspaces/workspace-a/runs/${acceptedRunId}/events`,
});
const posts = h => h.calls.filter(call => call.method === "POST");

for (const outcome of ["connection", "timeout", "invalid-json", "invalid-receipt"]) {
  test(`submission ${outcome} retains the key and immutable inputs for manual retry`, { timeout: 5000 }, async t => {
    let generated = 0;
    const h = await loadUi(t, { historical: true, randomUUID: () => { generated += 1; return requestKey; } });
    h.element("run-query").value = "original synthetic query";
    const pending = deferred();
    h.route(() => pending.promise);
    const creating = h.call("createRun");
    await until(() => posts(h).length === 1);
    assert.equal(h.element("create-run").disabled, true);
    await h.call("createRun");
    await h.call("retrySubmission");
    assert.equal(posts(h).length, 1);
    if (outcome === "timeout") h.clock.fire(30000);
    else if (outcome === "connection") pending.reject(new TypeError("Synthetic connection lost"));
    else if (outcome === "invalid-json") pending.resolve(new Response("{", { status: 202 }));
    else pending.resolve(json({ ...receipt(), events_url: "/wrong/events" }, 202));
    await creating;
    assert.equal(h.state.submission.accepted, null);
    assert.equal(h.element("create-run").disabled, false);
    assert.equal(h.element("retry-submission").disabled, false);
    assert.equal(posts(h).length, 1); // No automatic POST retry.
    h.element("run-query").value = "edited synthetic query";
    h.element("run-mode").value = "application";
    h.element("resume-document-id").value = nextRequestKey;
    h.element("run-form").reportValidity = () => false;
    h.route(() => failure(503));
    await h.call("retrySubmission");
    assert.equal(generated, 1);
    assert.equal(posts(h).length, 2);
    assert.deepEqual(posts(h).map(call => call.headers.get("Idempotency-Key")), [requestKey, requestKey]);
    assert.equal(posts(h)[0].body, posts(h)[1].body);
    assert.deepEqual(JSON.parse(posts(h)[1].body), {
      mode: "research", query: "original synthetic query", resume_document_id: null,
    });
    assert.equal(h.clock.pending.size, 0);
    pending.resolve(json(receipt(), 202));
  });
}

test("uncertain new submission requires confirmation and uses a fresh key and current inputs", { timeout: 5000 }, async t => {
  let confirmed = false, confirmations = 0, generated = 0;
  const h = await loadUi(t, {
    historical: true,
    randomUUID: () => [requestKey, nextRequestKey][generated++],
    confirm: () => { confirmations += 1; return confirmed; },
  });
  h.route(() => failure(503));
  h.element("run-query").value = "original";
  await h.call("createRun");
  h.element("run-query").value = "new input";
  await h.call("createRun");
  assert.equal(posts(h).length, 1);
  assert.equal(generated, 1);
  confirmed = true;
  await h.call("createRun");
  assert.equal(confirmations, 2);
  assert.equal(generated, 2);
  assert.equal(posts(h).length, 2);
  assert.equal(posts(h)[1].headers.get("Idempotency-Key"), nextRequestKey);
  assert.equal(JSON.parse(posts(h)[1].body).query, "new input");
});

test("409 disables retry without generating another request identity", { timeout: 5000 }, async t => {
  let generated = 0;
  const h = await loadUi(t, { historical: true, randomUUID: () => { generated += 1; return requestKey; } });
  h.route(() => failure(409));
  await h.call("createRun");
  assert.equal(h.state.submission.conflict, true);
  assert.equal(h.element("retry-submission").disabled, true);
  await h.call("retrySubmission");
  assert.equal(posts(h).length, 1);
  assert.equal(generated, 1);
  assert.match(h.element("submission-status").textContent, /Request conflict/);
});

for (const status of ["completed", "failed", "cancelled"]) {
  test(`${status} receipt reads current state before SSE and read recovery never POSTs`, { timeout: 5000 }, async t => {
    const h = await loadUi(t, { historical: true, randomUUID: () => requestKey });
    let reads = 0;
    const detail = deferred();
    h.route((url, options) => {
      if (options.method === "POST") return json(receipt(), 202);
      if (url.endsWith("/events")) return failure(404); // Stop the stream deterministically.
      reads += 1;
      return reads === 1 ? failure(503) : detail.promise;
    });
    await h.call("createRun");
    assert.equal(h.state.submission.accepted.run_id, acceptedRunId);
    assert.equal(h.state.run.status, undefined); // queued is only an acceptance snapshot.
    assert.equal(h.element("retry-run-read").hidden, false);
    assert.equal(h.calls.some(call => call.url.endsWith("/events")), false);
    await h.call("retrySubmission");
    const reading = h.call("readAcceptedRun");
    await until(() => reads === 2);
    assert.equal(h.calls.some(call => call.url.endsWith("/events")), false);
    detail.resolve(json(run(acceptedRunId, status)));
    await reading;
    await until(() => h.calls.some(call => call.url.endsWith("/events")));
    assert.equal(h.state.run.status, status);
    assert.equal(h.state.submission.readComplete, true);
    assert.equal(posts(h).length, 1);
    assert.deepEqual(h.calls.map(call => call.method || "GET"), ["POST", "GET", "GET", "GET"]);
  });
}

for (const change of ["workspace", "identity"]) {
  for (const boundary of ["POST", "GET"]) {
    test(`${change} switch rejects delayed submission ${boundary} callbacks`, { timeout: 5000 }, async t => {
      const h = await loadUi(t, { historical: true, randomUUID: () => requestKey });
      const pending = deferred();
      h.route((_url, options) => {
        if (boundary === "GET" && options.method === "POST") return json(receipt(), 202);
        return pending.promise;
      });
      const creating = h.call("createRun");
      await until(() => h.calls.some(call => call.method === boundary));
      if (change === "workspace") {
        h.call("renderWorkspaces", [{ workspace_id: "workspace-b", role: "admin", kind: "personal" }]);
      } else {
        h.route(() => json({ user_id: "actor-b", workspaces: [
          { workspace_id: "workspace-b", role: "admin", kind: "personal" },
        ] }));
        await h.call("loadMe");
      }
      h.state.run = run("new-context-run");
      pending.resolve(json(boundary === "POST" ? receipt() : run(acceptedRunId, "completed"), boundary === "POST" ? 202 : 200));
      await creating;
      assert.equal(h.state.submission, null);
      assert.equal(h.state.run.run_id, "new-context-run");
      assert.equal(h.element("create-run").disabled, false);
      assert.equal(h.calls.some(call => call.url.endsWith("/events")), false);
      await h.call("retrySubmission");
      assert.equal(posts(h).length, 1);
      assert.equal(h.clock.pending.size, 0);
    });
  }
}

test("missing secure UUID support prevents submission", { timeout: 5000 }, async t => {
  const h = await loadUi(t, { historical: true });
  await h.call("createRun");
  assert.equal(posts(h).length, 0);
  assert.equal(h.state.submission, null);
  assert.match(h.element("problem-panel").textContent, /Secure request ID unavailable/);
});

for (const operation of ["createRun", "retrySubmission", "submitRunIntent", "cancelCurrentRun", "submitDecision"]) {
  test(`retired production operation ${operation} cannot send a write`, async t => {
    const h = await loadUi(t);
    h.state.action = action();
    h.route(() => { throw new Error("Retired operations must not fetch"); });
    await h.call(operation, "approve", "synthetic");
    assert.equal(h.calls.length, 0);
    assert.match(h.element("problem-panel").textContent, /retired/);
  });
}

test("production renders historical runs and approvals without write controls", async t => {
  const h = await loadUi(t);
  h.call("renderSubmissionControls");
  assert.equal(h.element("create-run").disabled, true);
  assert.equal(h.element("retry-submission").disabled, true);
  h.call("renderRun", run());
  h.call("renderAction", action());
  assert.equal(h.element("run-panel").querySelector("button"), null);
  assert.equal(h.element("action-panel").querySelector("button"), null);
  assert.match(h.element("run-panel").textContent, /running/);
});
