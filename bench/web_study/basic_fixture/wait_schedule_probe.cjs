// Executes the actual runtime waiter in a controlled event scheduler. This
// counts predicate executions, not native CPU, elapsed time or model tokens.
const fs = require('node:fs');
const crypto = require('node:crypto');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const [sourcePath, outputPath] = process.argv.slice(2);
if (!sourcePath || !outputPath) throw new Error('usage: node wait_schedule_probe.cjs RUNTIME_JS NEW_OUTPUT_JSON');
const source = fs.readFileSync(sourcePath, 'utf8');
let next = 1, ready = false, observer, disconnected = false;
const frames = new Map(), timers = new Map();
const context = {
  Date, evaluations: 0,
  console: { debug() {} },
  document: { documentElement: {}, getElementById() { return ready ? {} : null; } },
  MutationObserver: class {
    constructor(callback) { observer = callback; }
    observe() {}
    disconnect() { disconnected = true; }
  },
  requestAnimationFrame(callback) { const id = next++; frames.set(id, callback); return id; },
  cancelAnimationFrame(id) { frames.delete(id); },
  setTimeout(callback) { const id = next++; timers.set(id, callback); return id; },
  clearTimeout(id) { timers.delete(id); },
  addEventListener() {}, removeEventListener() {},
};
context.window = context;
vm.createContext(context);
vm.runInContext(source, context);
context.greppyWaitForFunction('(++window.evaluations, document.getElementById("late") !== null)', 'schedule-proof', 60000, true);
const initial = context.evaluations;
for (let index = 0; index < 60; index++) {
  const frame = frames.entries().next().value;
  assert.ok(frame, 'an unchanged false DOM predicate continues to request frames');
  frames.delete(frame[0]);
  frame[1]();
}
const afterFrames = context.evaluations;
ready = true;
observer();
const slot = context['__greppyWait_schedule-proof'];
assert.equal(slot.status, 'ok');
assert.equal(slot.value, true);
assert.equal(frames.size, 0);
assert.equal(disconnected, true);
const report = {
  schema: 'greppy.wait-scheduler-source-probe.v1',
  source: sourcePath, source_sha256: crypto.createHash('sha256').update(source).digest('hex'),
  strict_boolean: true, unchanged_dom_frames: 60,
  initial_predicate_executions: initial, predicate_executions_after_frames: afterFrames,
  predicate_executions_after_mutation: context.evaluations,
  pending_frames_after_success: frames.size,
  evidence_scope: 'Actual JS waiter with synthetic frame/mutation scheduling; no native frame cadence, CPU, model-token, or elapsed-time claim.',
  efficiency_acceptance: false,
};
fs.writeFileSync(outputPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
process.stdout.write(JSON.stringify(report) + '\n');
