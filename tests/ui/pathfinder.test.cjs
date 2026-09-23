"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const historicalSource = readFileSync(join(__dirname, "../fixtures/legacy_ui/pathfinder.js"), "utf8");
const source = readFileSync(join(__dirname, "../../src/app/api/static/pathfinder.js"), "utf8");

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
  addEventListener(type, callback) { this.listeners.set(type, callback); }
  click() { assert.ok(this.listeners.has("click")); this.listeners.get("click")(); }
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
  const element = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  element("run-mode").value = "research";
  const clock = fakeClock();
  const calls = [];
  let route = () => { throw new Error("Unexpected fetch"); };
  const context = vm.createContext({
    document: { getElementById: element, createElement: tag => new Element(tag) },
    window: { setTimeout: clock.setTimeout, clearTimeout: clock.clearTimeout, confirm },
    crypto: { randomUUID },
    Headers, AbortController, DOMException, TextDecoder, Uint8Array, URL,
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
  return { state, clock, calls, element, call, route: handler => { route = handler; } };
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
