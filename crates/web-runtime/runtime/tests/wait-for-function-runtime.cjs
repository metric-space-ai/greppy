// Scheduler unit tests: execute the actual shipped waiter with deterministic
// browser scheduling seams. This does not replace the native Servo fixtures.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const source = fs.readFileSync(
  path.join(__dirname, "../js/wait-for-function-runtime.js"), "utf8",
);

function harness(expression = "ready", strictBoolean = false) {
  let nextId = 1;
  let mutation;
  let disconnected = 0;
  const frames = new Map();
  const deadlines = new Map();
  const listeners = new Map();
  const notices = [];
  const context = vm.createContext({
    ready: false,
    document: { documentElement: {} },
    console: { debug: (message) => notices.push(message) },
    MutationObserver: class {
      constructor(callback) { mutation = callback; }
      observe() {}
      disconnect() { disconnected++; }
    },
    requestAnimationFrame(callback) {
      const id = nextId++;
      frames.set(id, callback);
      return id;
    },
    cancelAnimationFrame(id) { frames.delete(id); },
    setTimeout(callback) {
      const id = nextId++;
      deadlines.set(id, callback);
      return id;
    },
    clearTimeout(id) { deadlines.delete(id); },
    addEventListener(name, callback) { listeners.set(name, callback); },
    removeEventListener(name) { listeners.delete(name); },
  });
  context.window = context;
  vm.runInContext(source, context);
  context.greppyWaitForFunction(expression, "scheduler-test", 1000, strictBoolean);
  return {
    context, frames, deadlines, listeners, notices,
    mutate: () => mutation(),
    disconnected: () => disconnected,
    expire() {
      const [id, callback] = Array.from(deadlines)[0];
      deadlines.delete(id);
      callback();
    },
    frame() {
      const pending = Array.from(frames);
      for (const [id, callback] of pending) {
        frames.delete(id);
        callback();
      }
    },
  };
}

test("mutation bursts keep exactly one pending animation frame", () => {
  const h = harness();
  assert.equal(h.frames.size, 1);
  for (let i = 0; i < 100; i++) h.mutate();
  assert.equal(h.frames.size, 1);
  for (let i = 0; i < 10; i++) {
    h.frame();
    h.mutate();
    assert.equal(h.frames.size, 1);
  }
  h.context.ready = true;
  h.mutate();
  assert.equal(h.frames.size, 0);
  assert.equal(h.deadlines.size, 0);
  assert.equal(h.listeners.size, 0);
  assert.equal(h.disconnected(), 1);
  assert.deepEqual(h.notices, ["__greppyWaitDone:scheduler-test:ok"]);
  h.frame();
  h.mutate();
  assert.equal(h.notices.length, 1);
});

test("timeout and predicate errors cancel all pending scheduling", () => {
  for (const outcome of ["timeout", "error"]) {
    const h = harness("if (ready) throw new Error('predicate-failure'); false");
    for (let i = 0; i < 10; i++) h.mutate();
    if (outcome === "timeout") {
      Array.from(h.deadlines.values())[0]();
    } else {
      h.context.ready = true;
      h.mutate();
    }
    assert.equal(h.frames.size, 0);
    assert.equal(h.deadlines.size, 0);
    assert.equal(h.listeners.size, 0);
    assert.equal(h.disconnected(), 1);
    assert.deepEqual(h.notices, ["__greppyWaitDone:scheduler-test:" + outcome]);
  }
});

test("immediate success neither schedules frames nor leaks a deadline", () => {
  const h = harness("({ answer: 42 })");
  assert.equal(h.frames.size, 0);
  assert.equal(h.deadlines.size, 0);
  assert.equal(h.listeners.size, 0);
  assert.equal(h.context["__greppyWait_scheduler-test"].value.answer, 42);
  assert.equal(h.context["__greppyWait_scheduler-test"].status, "ok");
});

test("property-only changes are still noticed on the next animation frame", () => {
  const h = harness();
  h.context.ready = true;
  // No mutation callback: DOM property updates need not mutate attributes.
  h.frame();
  assert.equal(h.frames.size, 0);
  assert.equal(h.deadlines.size, 0);
  assert.deepEqual(h.notices, ["__greppyWaitDone:scheduler-test:ok"]);
});

test("Boolean waits reject object truthiness including holds:false", () => {
  for (const source of ["({holds:false})", "({holds:true})", "'false'", "1", "null", "undefined"]) {
    const h = harness(source, true);
    const slot = h.context["__greppyWait_scheduler-test"];
    assert.equal(slot.status, "error");
    assert.match(slot.value, /INVALID_WAIT_PREDICATE/);
    assert.equal(h.frames.size, 0);
    assert.deepEqual(h.notices, ["__greppyWaitDone:scheduler-test:error"]);
  }
});

test("strict Boolean false, absent and AND remain pending until true", () => {
  for (const expression of ["ready", "!present", "ready && !present"]) {
    // `present` is introduced by the source itself to start every case false.
    const initial = expression.replaceAll("present", "!ready");
    const h = harness(initial, true);
    assert.equal(h.context["__greppyWait_scheduler-test"].done, 0);
    h.context.ready = true;
    h.mutate();
    assert.equal(h.context["__greppyWait_scheduler-test"].status, "ok");
    assert.equal(h.context["__greppyWait_scheduler-test"].value, true);
  }
});

test("strict source syntax errors return a bounded actionable category", () => {
  const h = harness("new RegExp('[bad')", true);
  const slot = h.context["__greppyWait_scheduler-test"];
  assert.equal(slot.status, "error");
  assert.equal(slot.value, "INVALID_WAIT_SOURCE: invalid JavaScript predicate or regular expression");
  assert.equal(h.frames.size, 0);
});

test("strict waits discard bookkeeping when the caller misses its final read", () => {
  for (const expression of ["true", "false", "({holds:false})"]) {
    const h = harness(expression, true);
    h.expire();
    assert.equal(h.context["__greppyWait_scheduler-test"], undefined);
    assert.equal(h.frames.size, 0);
    assert.equal(h.deadlines.size, 0);
    assert.equal(h.listeners.size, 0);
    assert.equal(h.notices.length, 1);
  }
});
