// Run the shipped waiter: a timeout must release state even when the controller
// never performs its final slot read. Native browser coverage remains separate.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const source = fs.readFileSync(
  path.join(__dirname, "../js/wait-for-function-runtime.js"), "utf8",
);

function fixture(expression, strictBoolean) {
  let nextTimer = 1;
  let disconnected = 0;
  const timers = new Map();
  const listeners = new Map();
  const notices = [];
  const context = vm.createContext({
    document: { documentElement: {} },
    console: { debug: (value) => notices.push(value) },
    MutationObserver: class {
      observe() {}
      disconnect() { disconnected++; }
    },
    setTimeout(callback) {
      const id = nextTimer++;
      timers.set(id, callback);
      return id;
    },
    clearTimeout(id) { timers.delete(id); },
    addEventListener(name, callback) { listeners.set(name, callback); },
    removeEventListener(name) { listeners.delete(name); },
  });
  context.window = context;
  vm.runInContext(source, context);
  context.greppyWaitForFunction(expression, "slot-contract", 1000, strictBoolean);
  return {
    context, timers, listeners, notices,
    disconnected: () => disconnected,
    expire() {
      // The production waiter registers its deadline before its polling timer.
      const callback = timers.values().next().value;
      assert.equal(typeof callback, "function");
      callback();
    },
  };
}

test("ordinary and Boolean predicate timeouts release their slots and scheduling", () => {
  for (const strictBoolean of [false, true]) {
    const h = fixture("false", strictBoolean);
    assert.equal(h.context["__greppyWait_slot-contract"].done, 0);
    h.expire();
    assert.equal(h.context["__greppyWait_slot-contract"], undefined,
      `timeout slot retained with strictBoolean=${strictBoolean}`);
    assert.equal(h.timers.size, 0);
    assert.equal(h.listeners.size, 0);
    assert.equal(h.disconnected(), 1);
    assert.deepEqual(h.notices, ["__greppyWaitDone:slot-contract:timeout"]);
  }
});

test("ordinary success preserves the value for the controller read", () => {
  const h = fixture("({ answer: 42 })", false);
  const slot = h.context["__greppyWait_slot-contract"];
  assert.equal(slot.done, 1);
  assert.equal(slot.status, "ok");
  assert.equal(slot.value.answer, 42);
  assert.equal(h.timers.size, 0);
  assert.equal(h.listeners.size, 0);
  assert.equal(h.disconnected(), 1);
  assert.deepEqual(h.notices, ["__greppyWaitDone:slot-contract:ok"]);
});
