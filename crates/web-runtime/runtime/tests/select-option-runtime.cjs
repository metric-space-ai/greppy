// Executes the shipped helper against a minimal select value-setter seam.
// Native Servo/CLI tests remain necessary for browser and protocol acceptance.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const context = vm.createContext({
  Event: class { constructor(type, options) { this.type = type; this.bubbles = options.bubbles; } },
});
for (const file of ['../../../web-client/src/select-choices.js', '../js/select-option-runtime.js']) {
  vm.runInContext(fs.readFileSync(path.join(__dirname, file), 'utf8'), context);
}
function fixture(attributes = {}) {
  const options = [
    { value: 'descending', label: 'High to low', selected: true },
    { value: 'ascending', label: 'Low to high', selected: false },
    { value: '', label: 'No sorting', selected: false },
  ].map(item => ({ ...item, disabled: false, parentElement: null }));
  const node = {
    tagName: 'SELECT', options, disabled: false, events: [], assignments: 0,
    getAttribute(name) { return attributes[name] ?? null; },
    matches(selector) { assert.equal(selector, ':disabled'); return false; },
    get selectedOptions() { return options.filter(option => option.selected); },
    get value() { return this.selectedOptions[0]?.value ?? ''; },
    set value(value) {
      this.assignments++;
      if (this.ignoreWrites) return;
      const first = options.find(option => option.value === value);
      for (const option of options) option.selected = option === first;
    },
    dispatchEvent(event) {
      assert.equal(event.bubbles, true);
      this.events.push(event.type);
      if (event.type === 'change' && this.onChange) this.onChange();
    },
  };
  return node;
}
const select = (node, value) => context.greppySelectOption(node, value);

test('unknown values and labels fail before mutation with bounded fenced choices', () => {
  for (const unknown of ['low', 'Low to high']) {
    const node = fixture();
    assert.throws(() => select(node, unknown), error => {
      assert.match(error.message, /OPTION_NOT_FOUND/);
      assert.match(error.message, /greppy web select TARGET VALUE/);
      assert.match(error.message, /UNTRUSTED_PAGE_CONTENT_BEGIN\n/);
      assert.match(error.message, /"value":"ascending","label":"Low to high"/);
      assert.match(error.message, /\nUNTRUSTED_PAGE_CONTENT_END$/);
      return true;
    });
    assert.equal(node.value, 'descending');
    assert.equal(node.assignments, 0);
    assert.deepEqual(node.events, []);
  }
});

test('valid value notifies the application once and same-value repeat is a no-op', () => {
  const node = fixture();
  assert.equal(select(node, 'ascending'), true);
  assert.equal(node.value, 'ascending');
  assert.equal(node.selectedOptions.length, 1);
  assert.equal(node.assignments, 1);
  assert.deepEqual(node.events, ['input', 'change']);
  assert.equal(select(node, 'ascending'), true);
  assert.equal(node.assignments, 1);
  assert.deepEqual(node.events, ['input', 'change']);
});

test('empty value selects its actual option instead of clearing all options', () => {
  const node = fixture();
  assert.equal(select(node, ''), true);
  assert.equal(node.value, '');
  assert.equal(node.selectedOptions.length, 1);
  assert.equal(node.selectedOptions[0], node.options[2]);
});

test('disabled option, optgroup, select and inherited disabling refuse without writes', () => {
  for (const disable of [
    node => { node.options[1].disabled = true; },
    node => { node.options[1].parentElement = { tagName: 'OPTGROUP', disabled: true, getAttribute() { return null; } }; },
    node => { node.disabled = true; },
    node => { node.matches = () => true; },
  ]) {
    const node = fixture();
    disable(node);
    assert.throws(() => select(node, 'ascending'), /OPTION_DISABLED/);
    assert.equal(node.assignments, 0);
    assert.deepEqual(node.events, []);
  }
});

test('a failed setter does not dispatch events or certify success', () => {
  const node = fixture();
  node.ignoreWrites = true;
  assert.throws(() => select(node, 'ascending'), /SELECTION_NOT_APPLIED/);
  assert.equal(node.assignments, 1);
  assert.equal(node.value, 'descending');
  assert.deepEqual(node.events, []);
});

test('page-handler reversal fails loudly without rollback or replay', () => {
  const node = fixture();
  node.onChange = () => { node.value = 'descending'; };
  assert.throws(() => select(node, 'ascending'), /SELECTION_CHANGED/);
  assert.equal(node.value, 'descending');
  assert.equal(node.assignments, 2);
  assert.deepEqual(node.events, ['input', 'change']);
});

test('validation searches beyond the diagnostic preview', () => {
  const node = fixture();
  for (let index = 0; index < 20; index++) {
    node.options.push({ value: `extra-${index}`, label: `Extra ${index}`, selected: false });
  }
  assert.equal(select(node, 'extra-19'), true);
  assert.equal(node.value, 'extra-19');
});

test('sensitive diagnostics never reveal available or requested secrets', () => {
  const node = fixture({ autocomplete: 'one-time-code' });
  node.options[1].value = 'private-available-code';
  assert.throws(() => select(node, 'private-requested-code'), error => {
    assert.match(error.message, /OPTION_NOT_FOUND/);
    assert.doesNotMatch(error.message, /private-|Low to high|select_choices/);
    return true;
  });
  assert.equal(node.assignments, 0);
});

test('wrong element and non-string values fail without coercion', () => {
  assert.throws(() => select({ tagName: 'INPUT' }, 'ascending'), /INVALID_SELECT_TARGET/);
  for (const value of [null, false, 1, ['ascending'], { label: 'Low to high' }]) {
    const node = fixture();
    assert.throws(() => select(node, value), /INVALID_OPTION_VALUE/);
    assert.equal(node.assignments, 0);
  }
});
