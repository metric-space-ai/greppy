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

function harness(expression = "ready") {
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
  context.greppyWaitForFunction(expression, "scheduler-test", 1000);
  return {
    context, frames, deadlines, listeners, notices,
    mutate: () => mutation(),
    disconnected: () => disconnected,
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
