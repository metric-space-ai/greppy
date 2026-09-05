// Executes the shared descriptor and projection as one expression, without a
// browser. Actual CLI/Servo binding still needs its native integration proof.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const helper = fs.readFileSync(path.join(__dirname, '../src/select-choices.js'), 'utf8');
const source = fs.readFileSync(path.join(__dirname, '../src/describe-node.js'), 'utf8');
const context = vm.createContext({ getComputedStyle: () => ({ visibility: 'visible', display: 'block' }) });
const describe = vm.runInContext(`(function() {\n${helper}\nreturn ${source};\n})()`, context);
function element(extra = {}) {
  return { tagName: 'SELECT', id: 'order', textContent: 'UnsortedLow to highHigh to low',
    value: '', disabled: false, attributes: [], options: [],
    getAttribute() { return null; }, matches() { return false; },
    getBoundingClientRect() { return { x: 1, y: 2, width: 100, height: 20 }; }, ...extra };
}
function describeNode(node) { return JSON.parse(JSON.stringify(describe(node, true))); }

test('inspect exposes usable option values with labels and disabled group state', () => {
  const node = element({ options: [
    { value: '', label: 'Unsorted', disabled: false, parentElement: null },
    { value: 'ascending', label: 'Low to high', disabled: false, parentElement: null },
    { value: 'descending', label: 'High to low', disabled: false,
      parentElement: { tagName: 'OPTGROUP', disabled: true, getAttribute() { return null; } } },
  ] });
  const before = JSON.stringify(node);
  const result = describeNode(node);
  assert.equal(result.id, 'order');
  assert.equal(result.value, '');
  assert.equal(result.select_choices.schema, 'greppy.web.select-choices.v1');
  assert.deepEqual(result.select_choices.choices.map(({ value, label, disabled }) => ({ value, label, disabled })), [
    { value: '', label: 'Unsorted', disabled: false },
    { value: 'ascending', label: 'Low to high', disabled: false },
    { value: 'descending', label: 'High to low', disabled: true },
  ]);
  assert.equal(result.select_choices.choices_total, 3);
  assert.equal(result.select_choices.choices_truncated, false);
  assert.equal(JSON.stringify(node), before);
});

test('non-select state is retained without irrelevant choice data', () => {
  const result = describeNode(element({ tagName: 'INPUT', checked: false, disabled: true, value: '3' }));
  assert.equal(result.checked, false);
  assert.equal(result.disabled, true);
  assert.equal(result.value, '3');
  assert.equal(Object.hasOwn(result, 'select_choices'), false);
});

test('sensitive select descriptor never enumerates its option collection', () => {
  const node = element({ getAttribute(name) { return name === 'autocomplete' ? 'one-time-code' : null; } });
  Object.defineProperty(node, 'options', { get() { throw new Error('must not enumerate'); } });
  assert.equal(Object.hasOwn(describeNode(node), 'select_choices'), false);
});
