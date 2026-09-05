// Projection tests run the actual helper without launching a browser. Native
// selection and descriptor integration require their separate Servo tests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../js/select-choices-runtime.js'), 'utf8');
const context = vm.createContext({});
vm.runInContext(source, context);

function select(options, attributes = {}) {
  return {
    tagName: 'SELECT', options, disabled: false,
    getAttribute(name) { return attributes[name] ?? null; },
    matches(selector) { assert.equal(selector, ':disabled'); return false; },
  };
}
function option(value, label, extra = {}) {
  return { value, label, disabled: false, parentElement: null, ...extra };
}
function snapshot(node) {
  return JSON.parse(JSON.stringify(context.greppySelectChoicesSnapshot(node)));
}

test('real values and labels remain distinct, including empty and duplicate values', () => {
  const node = select([
    option('ascending', 'Low to high'), option('', 'No sorting'), option('ascending', 'Alias'),
  ]);
  const before = JSON.stringify(node.options);
  const result = snapshot(node);
  assert.equal(result.schema, 'greppy.web.select-choices.v1');
  assert.equal(result.choices_total, 3);
  assert.equal(result.choices_truncated, false);
  assert.deepEqual(result.choices.map(item => [item.value, item.label]), [
    ['ascending', 'Low to high'], ['', 'No sorting'], ['ascending', 'Alias'],
  ]);
  assert.ok(result.choices.every(item => !item.disabled && !item.value_truncated && !item.label_truncated));
  assert.equal(JSON.stringify(node.options), before);
});

test('large collections inspect at most eight options and retain the full count', () => {
  const touched = [];
  const options = new Proxy({ length: 1000000 }, {
    get(target, name) {
      if (name === 'length') return target.length;
      const index = Number(name);
      assert.ok(index >= 0 && index < 8, `read beyond bounded prefix: ${name}`);
      touched.push(index);
      return option(String(index), `Label ${index}`);
    },
  });
  const result = snapshot(select(options));
  assert.deepEqual(touched, [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.equal(result.choices.length, 8);
  assert.equal(result.choices_total, 1000000);
  assert.equal(result.choices_truncated, true);
});

test('long values are unavailable rather than fake actionable prefixes', () => {
  const result = snapshot(select([
    option('x'.repeat(160), 'short'), option('x'.repeat(161), 'y'.repeat(161)),
  ]));
  assert.equal(result.choices[0].value.length, 160);
  assert.equal(result.choices[0].value_truncated, false);
  assert.equal(result.choices[1].value, null);
  assert.equal(result.choices[1].value_truncated, true);
  assert.equal(result.choices[1].label, 'y'.repeat(160));
  assert.equal(result.choices[1].label_truncated, true);
});

test('Unicode limits do not split surrogate pairs or reject exactly 160 characters', () => {
  const result = snapshot(select([
    option('🌍'.repeat(160), '🌍'.repeat(160)), option('🌍'.repeat(161), '🌍'.repeat(161)),
  ]));
  assert.equal(result.choices[0].value, '🌍'.repeat(160));
  assert.equal(result.choices[0].value_truncated, false);
  assert.equal(result.choices[1].value, null);
  assert.equal(result.choices[1].label, '🌍'.repeat(160));
  assert.equal(result.choices[1].label_truncated, true);
});

test('disabled option, optgroup, select and inherited disabled state are retained', () => {
  const group = { tagName: 'OPTGROUP', disabled: true, getAttribute() { return null; } };
  const node = select([
    option('one', 'One'), option('two', 'Two', { disabled: true }),
    option('three', 'Three', { parentElement: group }),
  ]);
  assert.deepEqual(snapshot(node).choices.map(item => item.disabled), [false, true, true]);
  node.disabled = true;
  assert.ok(snapshot(node).choices.every(item => item.disabled));
  node.disabled = false;
  node.matches = () => true; // e.g. a disabled fieldset, as resolved by the DOM
  assert.ok(snapshot(node).choices.every(item => item.disabled));
  const aria = select([option('one', 'One')], { 'aria-disabled': 'true' });
  assert.equal(snapshot(aria).choices[0].disabled, true);
});

test('sensitive selects never enumerate options; non-selects emit no choices', () => {
  for (const token of ['current-password', 'new-password', 'one-time-code', 'cc-number', 'cc-csc']) {
    const node = select(null, { autocomplete: `section-test ${token.toUpperCase()}` });
    Object.defineProperty(node, 'options', { get() { throw new Error('secret option read'); } });
    assert.equal(snapshot(node), null);
  }
  assert.equal(snapshot(null), null);
  assert.equal(snapshot({ tagName: 'INPUT' }), null);
});

test('empty lists are complete and page strings remain inert data', () => {
  assert.deepEqual(snapshot(select([])), {
    schema: 'greppy.web.select-choices.v1', choices: [], choices_total: 0, choices_truncated: false,
  });
  const hostile = '"; globalThis.injectionRan = true; //';
  assert.equal(snapshot(select([option(hostile, hostile)])).choices[0].value, hostile);
  assert.equal(context.injectionRan, undefined);
});
