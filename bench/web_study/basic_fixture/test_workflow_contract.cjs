const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../../..');
const conditionSource = fs.readFileSync(path.join(root, 'crates/web-client/src/workflow-condition.js'), 'utf8');
const preflightSource = fs.readFileSync(path.join(root, 'crates/web-client/src/workflow-preflight.js'), 'utf8');
function setup() {
  const calls = {live:0, detached:[], xpath:[], executed:false};
  const context = vm.createContext({document:{
    title:'Saved',
    createDocumentFragment() { return {querySelectorAll(value) { calls.detached.push(value); if (value === '[') throw new SyntaxError('bad selector'); return []; }}; },
    createExpression(value) { calls.xpath.push(value); if (value === 'bad(') throw new SyntaxError('bad xpath'); return {}; },
    querySelectorAll(value) { calls.live++; return value === '#done' ? [{}] : []; },
  }, location:{href:'https://fixture.test/done'}, marker() { calls.executed=true; }});
  return {calls, condition:vm.runInContext(conditionSource, context), preflight:vm.runInContext(preflightSource, context)};
}

test('preflight validates later selectors without observing or mutating the live document', () => {
  const x = setup();
  const result = x.preflight([{selector:{type:'css',value:'#button'}}, {condition:{query:'css=['}}], x.condition);
  assert.equal(result.valid, false); assert.equal(result.step, 2);
  assert.equal(x.calls.live, 0); assert.equal(x.calls.executed, false);
  assert.deepEqual(x.calls.detached, ['#button','[']);
});

test('arbitrary text stays data; regex and XPath syntax failures are not false conditions', () => {
  const x = setup();
  assert.equal(x.preflight([{condition:{query:'text=marker()'}}], x.condition).valid, true);
  assert.equal(x.calls.executed, false);
  assert.equal(x.preflight([{condition:{query:'text~/[/i',absent:true}}], x.condition).valid, false);
  assert.equal(x.preflight([{condition:{query:'xpath=bad(',absent:true}}], x.condition).valid, false);
  assert.equal(x.preflight([{condition:{url:'~/[/i',absent:true}}], x.condition).valid, false);
  assert.equal(x.calls.live, 0);
});

test('presence, conjunction and absence produce strict booleans without repeated preflight', () => {
  const x = setup();
  assert.equal(x.condition({query:'css=#done', url:'~/done$/', title:'Saved'}, null, false), true);
  assert.equal(x.condition({query:'css=#missing', absent:true}, null, false), true);
  assert.equal(x.condition({query:'css=#done', title:'Other', absent:false}, null, false), false);
  assert.deepEqual(x.calls.detached, []);
});

test('missing reference bindings cannot prove absence', () => {
  const x = setup();
  assert.throws(() => x.condition({query:'@7',absent:true}, null, false), /REF_CONDITION_UNSUPPORTED/);
  assert.equal(x.condition({query:'@7',absent:false}, [{}], false), true);
});
