// Component regression: actual condition guard + actual native identity check
// and WeakMap registry. DOM lookup is a controlled fixture, not a browser gate.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const src = path.join(__dirname, '../src');
const worker = fs.readFileSync(path.join(src, 'content_worker.rs'), 'utf8');
const match = worker.match(/function greppyObservedRefMatches\(selector, nodes\) \{[\s\S]*?\n\}/);
assert.ok(match, 'must test the actual native locator identity check');
const guard = fs.readFileSync(path.join(src, 'observed-ref-condition.js'), 'utf8');
const registry = fs.readFileSync(path.join(src, 'observed-ref-registry.js'), 'utf8');

function fixture() {
  const world = vm.createContext({});
  vm.runInContext(`
    let snapshot = 'document-one';
    let document = {documentElement: {getAttribute: () => snapshot}};
    const makeRegistry = (${registry});
    const window = {__greppyObservedRefs: makeRegistry(document, null, snapshot, 1, 200)};
    const original = {ownerDocument: document, isConnected: true};
    const reference = window.__greppyObservedRefs.refFor(original);
    const selector = {snapshot, observed_ref: reference};
    let visibleNodes = [original];
    function greppyResolveNodes() { return visibleNodes; }
    ${match[0]}
    const condition = (${guard});
    let evaluated = 0;
    function sample(absent = false) {
      return condition(selector, nodes => { evaluated++; return (nodes.length > 0) !== absent; });
    }
  `, world);
  return expression => vm.runInContext(expression, world);
}

test('active refs evaluate without becoming CSS and absence is false', () => {
  const run = fixture();
  assert.equal(run('sample()'), true);
  assert.equal(run('sample(true)'), false);
  assert.equal(run('evaluated'), 2);
});

for (const [name, mutation] of [
  ['removed node', 'original.isConnected = false; visibleNodes = []'],
  ['replacement with copied attributes', 'visibleNodes = [{ownerDocument: document, isConnected: true}]'],
  ['another document', 'document = {documentElement: {getAttribute: () => snapshot}}'],
  ['another tab snapshot', "snapshot = 'tab-two'"],
  ['unknown ref', 'selector.observed_ref = 199'],
  ['missing observation registry', 'window.__greppyObservedRefs = null'],
]) {
  test(`${name} throws STALE_REF before --absent can invert anything`, () => {
    const run = fixture();
    assert.equal(run('sample()'), true);
    run(mutation);
    assert.throws(() => run('sample(true)'), /STALE_REF/);
    assert.equal(run('evaluated'), 1, 'stale sample must not execute the condition body');
  });
}

test('every repeated sample checks identity again', () => {
  const run = fixture();
  assert.equal(run('sample()'), true);
  assert.equal(run('sample()'), true);
  run('original.isConnected = false');
  assert.throws(() => run('sample()'), /STALE_REF/);
  assert.equal(run('evaluated'), 2);
});
