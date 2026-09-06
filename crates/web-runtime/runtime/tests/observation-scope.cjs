const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/observation-scope.js'), 'utf8');
const context = vm.createContext({});
vm.runInContext(source, context);

function fixture() {
  const document = {};
  function element(name, parent = null, visible = true) {
    return { name, parentElement: parent, ownerDocument: document, nodeType: 1, visible,
      contains(other) {
        for (let node = other; node; node = node.parentElement) if (node === this) return true;
        return false;
      }
    };
  }
  const body = element('body');
  const background = element('background', body);
  const dialog = element('dialog', body);
  const field = element('field', dialog);
  const nested = element('nested', dialog);
  const button = element('button', nested);
  return { document, element, body, background, dialog, field, nested, button };
}
function scope(f, resolved, limit = 20) {
  return context.greppyObservationScope(f.document, 'role=dialog', resolved, n => n.visible, limit);
}

test('a selected container includes itself and descendants, never background', () => {
  const f = fixture();
  const result = scope(f, [f.dialog]);
  for (const node of [f.dialog, f.field, f.nested, f.button]) assert.equal(result.includes(node), true);
  for (const node of [f.body, f.background]) assert.equal(result.includes(node), false);
  assert.deepEqual(Array.from(result.collect([f.background, f.field, f.button], 20).nodes), [f.field, f.button]);
});

test('absent or hidden dialog never falls back to the whole page', () => {
  const f = fixture();
  f.dialog.visible = false;
  for (const resolved of [[], [f.dialog]]) {
    const result = scope(f, resolved);
    assert.equal(result.roots.length, 0);
    assert.equal(result.metadata.visible_matches, 0);
    assert.equal(result.includes(f.background), false);
    assert.equal(result.collect([f.background, f.field], 20).nodes.length, 0);
  }
  assert.equal(scope(f, [f.dialog]).metadata.matched_elements, 1);
});

test('nested and duplicate roots are deduplicated before applying the bound', () => {
  const f = fixture();
  const other = f.element('other', f.body);
  const result = scope(f, [f.nested, f.dialog, f.dialog, other], 2);
  assert.deepEqual(Array.from(result.roots), [f.dialog, other]);
  assert.equal(result.metadata.matched_elements, 3);
  assert.equal(result.metadata.roots_total, 2);
  assert.equal(result.metadata.roots_truncated, false);
});

test('multiple disjoint roots stay separate and truncation is explicit', () => {
  const f = fixture();
  const other = f.element('other', f.body);
  const extra = f.element('extra', f.body);
  const result = scope(f, [f.dialog, other, extra], 2);
  assert.equal(result.includes(other), true);
  assert.equal(result.includes(extra), false);
  assert.equal(result.metadata.roots_total, 3);
  assert.equal(result.metadata.roots_returned, 2);
  assert.equal(result.metadata.roots_truncated, true);
});

test('item cap applies after membership and visibility with exact counts', () => {
  const f = fixture();
  const hidden = f.element('hidden', f.dialog, false);
  const result = scope(f, [f.dialog]).collect([f.background, hidden, f.field, f.button], 1);
  assert.deepEqual(Array.from(result.nodes), [f.field]);
  assert.equal(result.total, 2);
  assert.equal(result.truncated, true);
});

test('non-elements, cross-document nodes and malformed bounds fail explicitly', () => {
  const f = fixture();
  for (const node of [null, { nodeType: 3, ownerDocument: f.document }, fixture().dialog]) {
    assert.throws(() => scope(f, [node]), /current document/);
  }
  for (const limit of [0, -1, 1.5, Infinity]) assert.throws(() => scope(f, [], limit), /positive integer/);
  assert.throws(() => scope(f, []).collect([], -1), /nonnegative integer/);
  assert.throws(() => context.greppyObservationScope(f.document, ' ', [], () => true, 20), /empty/);
});

test('selection does not mutate source nodes, input order or document', () => {
  const f = fixture();
  const resolved = [f.dialog, f.nested];
  const before = resolved.slice();
  const keys = Object.keys(f.dialog);
  const result = scope(f, resolved);
  assert.deepEqual(resolved, before);
  assert.deepEqual(Object.keys(f.dialog), keys);
  assert.deepEqual(f.document, {});
  assert.equal(result.roots[0], f.dialog);
});
