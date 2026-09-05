// Prospective integration test against an explicitly supplied patched runtime
// source. It executes the real OBSERVE_JS template with DOM-shaped objects;
// native refs/layout and compiled Servo integration are not tested here.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
if (!process.env.GREPPY_OBSERVE_PROBE_SOURCE) throw new Error('set GREPPY_OBSERVE_PROBE_SOURCE to the patched content_worker.rs');
const runtime = fs.readFileSync(process.env.GREPPY_OBSERVE_PROBE_SOURCE, 'utf8');
const start = 'const OBSERVE_JS: &str = r#"';
const from = runtime.indexOf(start);
assert.ok(from >= 0, 'actual Observe template exists');
const to = runtime.indexOf('"#;', from + start.length);
assert.ok(to > from);
const helper = fs.readFileSync(path.join(__dirname, '../../../crates/web-client/src/select-choices.js'), 'utf8');
assert.ok(runtime.includes('.replace("__GREPPY_SELECT_CHOICES__", greppy_web_client::SELECT_CHOICES_JS)'), 'Rust builder injects the shared helper');
const template = runtime.slice(from + start.length, to)
  .replace('__GREPPY_NATIVE_LABEL_TEXT__', 'function greppyNativeLabelText(label) { return label.textContent; }')
  .replace('__GREPPY_SELECT_CHOICES__', helper)
  .replace('__GREPPY_REF_REGISTRY__', '(function() { return { snapshot: "probe", refFor: function() { return 1; } }; })')
  .replace('__GREPPY_REF_LIMIT__', '100')
  .replace('__GREPPY_REF_FIRST__', '1').replace('__GREPPY_REF_LAST__', '100')
  .replace('__GREPPY_SNAPSHOT__', '"probe"');
assert.equal(/__GREPPY_/.test(template), false, 'every template slot bound');
function node(extra = {}) {
  return { tagName: 'SELECT', type: 'select-one', value: '', labels: [], disabled: false,
    innerText: '', textContent: '', validity: { valid: true }, options: [],
    getAttribute(name) { return name === 'aria-label' ? 'Price order' : null; },
    setAttribute() {}, hasAttribute() { return false; }, matches() { return false; },
    getBoundingClientRect() { return { width: 120, height: 20 }; }, ...extra };
}
function observe(element) {
  const document = { title: 'Fixture', body: { innerText: 'Fixture' }, documentElement: { setAttribute() {} },
    querySelectorAll(selector) { return selector.startsWith('a[href],button,input,select') ? [element] : []; },
    getElementById() { return null; } };
  return JSON.parse(vm.runInNewContext(template, { document, window: {}, location: { href: 'https://fixture.invalid/' },
    getComputedStyle() { return { display: 'block', visibility: 'visible' }; } }));
}
test('first Observe includes actionable values before any failed Select', () => {
  const result = observe(node({ options: [
    { value: '', label: 'Unsorted', selected: true, disabled: false },
    { value: 'ascending', label: 'Low to high', selected: false, disabled: false },
    { value: 'descending', label: 'High to low', selected: false, disabled: false,
      parentElement: { tagName: 'OPTGROUP', disabled: true, getAttribute() { return null; } } },
  ] }));
  assert.equal(result.actionable_schema, 'greppy.web.actionable.v2');
  const control = result.actionables[0];
  assert.equal(control.ref, '@1');
  assert.deepEqual(control.selected_options, [{ value: '', label: 'Unsorted' }]);
  assert.deepEqual(control.select_choices.choices.map(({ value, label, disabled }) => ({ value, label, disabled })), [
    { value: '', label: 'Unsorted', disabled: false },
    { value: 'ascending', label: 'Low to high', disabled: false },
    { value: 'descending', label: 'High to low', disabled: true },
  ]);
  assert.equal(control.select_choices.choices_total, 3);
  assert.equal(control.select_choices.choices_truncated, false);
});
test('non-select actionables do not acquire a choices field', () => {
  const control = observe(node({ tagName: 'INPUT', type: 'checkbox', checked: false })).actionables[0];
  assert.equal(control.checked, false);
  assert.equal(Object.hasOwn(control, 'select_choices'), false);
});
test('sensitive select is redacted and never enumerated by Observe', () => {
  const element = node({ value: 'secret', getAttribute(name) { return name === 'autocomplete' ? 'one-time-code' : null; } });
  Object.defineProperty(element, 'options', { get() { throw new Error('sensitive options accessed'); } });
  const control = observe(element).actionables[0];
  assert.equal(control.value_redacted, true);
  assert.equal(control.value, null);
  assert.equal(control.text, '');
  assert.equal(Object.hasOwn(control, 'select_choices'), false);
});
