const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../src/observed-working-scope.js'), 'utf8'), context);

function element(tag, parent = null, role = null) {
  return {
    tagName: tag, parentElement: parent, innerText: '',
    getAttribute: key => key === 'role' ? role : null,
    contains(other) {
      for (let node = other; node; node = node.parentElement) if (node === this) return true;
      return false;
    }
  };
}
function plan({ native = [], declared = [], focus = null, unsupported = false } = {}) {
  const document = {
    activeElement: focus, body: {}, documentElement: {},
    querySelectorAll(selector) {
      if (selector === ':modal') {
        if (unsupported) throw new SyntaxError('unsupported selector');
        return native;
      }
      assert.equal(selector, '[aria-modal="true"]');
      return declared;
    }
  };
  return context.greppyWorkingScopePlan(document, node => !node.hidden);
}

test('unrelated unfocused modals remain ambiguous instead of trusting DOM order', () => {
  const result = plan({ native: [element('DIALOG'), element('DIALOG')] });
  assert.equal(result.scope, null);
  assert.equal(result.ambiguous, true);
  assert.equal(result.candidateCount, 2);
});

test('nested focused native modal chooses the innermost container', () => {
  const outer = element('DIALOG');
  const inner = element('DIALOG', outer);
  const focus = element('INPUT', inner);
  const result = plan({ native: [outer, inner], focus });
  assert.equal(result.scope, inner);
  assert.equal(result.provenance, 'native_modal');
});

test('declared modality does not displace a native modal', () => {
  const native = element('DIALOG');
  const declared = element('DIV', null, 'dialog');
  const result = plan({ native: [native], declared: [declared], focus: element('INPUT', declared) });
  assert.equal(result.scope, native);
  assert.equal(result.provenance, 'native_modal');
});

test('unsupported native selector reports uncertainty, not open-dialog modality', () => {
  const openDialog = element('DIALOG');
  const result = plan({ unsupported: true, focus: element('INPUT', openDialog) });
  assert.equal(result.scope, null);
  assert.equal(result.nativeSupported, false);
  assert.equal(result.provenance, null);
});

test('ARIA-only scope explicitly remains a declaration and ignores hidden candidates', () => {
  const declared = element('DIV', null, 'dialog');
  const hidden = element('DIV', null, 'dialog');
  hidden.hidden = true;
  const result = plan({ declared: [hidden, declared] });
  assert.equal(result.scope, declared);
  assert.equal(result.provenance, 'declared_aria_modal');
  assert.equal(result.candidateCount, 1);
});

test('ancestry and the extra reference reservation stay bounded', () => {
  const root = element('DIALOG');
  let parent = root;
  for (let i = 0; i < 20; i++) parent = element('DIV', parent, 'form');
  const result = plan({ native: [root], focus: element('INPUT', parent) });
  assert.equal(result.ancestry.length, 8);
  assert.equal(result.ancestryTruncated, true);
  assert.equal(result.extraNodes.length, 10);
  assert.ok(result.extraNodes.includes(root));
});

test('snapshot selects foreground refs without deleting background records', () => {
  const root = element('DIALOG');
  const focus = element('INPUT', root);
  const background = element('BUTTON');
  const candidates = [background, focus];
  const actionables = [{ ref: '@1' }, { ref: '@2' }];
  const result = context.greppyWorkingScopeSnapshot(
    plan({ native: [root], focus }), candidates, candidates, actionables,
    node => ({ ref: node === root ? '@3' : '@2', role: 'dialog', name: 'Reserve' })
  );
  assert.equal(result.actionable_refs.join(','), '@2');
  assert.equal(result.background_count, 1);
  assert.equal(result.background_returned, 1);
  assert.equal(result.focus_ref, '@2');
  assert.equal(actionables.length, 2);
  assert.equal(candidates.length, 2);
});
