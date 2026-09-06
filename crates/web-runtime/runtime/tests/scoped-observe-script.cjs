// Execute the production OBSERVE_JS and its production helpers against a
// controlled DOM. This checks composition, not native-engine acceptance.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const src = path.join(__dirname, '../src');
const client = path.join(__dirname, '../../../web-client/src');
const worker = fs.readFileSync(path.join(src, 'content_worker.rs'), 'utf8');
const script = worker.match(/const OBSERVE_JS: &str = r#"([\s\S]*?)"#;/)?.[1];
assert.ok(script, 'production observation script must exist');
function read(base, name) { return fs.readFileSync(path.join(base, name), 'utf8'); }
const substitutions = {
  __GREPPY_NATIVE_LABEL_TEXT__: read(src, 'native-label-text.js'),
  __GREPPY_SELECT_CHOICES__: read(client, 'select-choices.js'),
  __GREPPY_WORKING_SCOPE__: read(src, 'observed-working-scope.js'),
  __GREPPY_OBSERVATION_SCOPE__: read(src, 'observation-scope.js'),
  __GREPPY_QUERY_RESOLVER__: read(client, 'node-query-resolver.js'),
  __GREPPY_REF_REGISTRY__: read(src, 'observed-ref-registry.js'),
  __GREPPY_REF_LIMIT__: '200',
};
function fixture() {
  const nodes = [];
  const document = { title: 'Page title', activeElement: null };
  function element(tag, id, text, parent = null, attributes = {}) {
    const attrs = { id, ...attributes };
    const node = {
      nodeType: 1, tagName: tag.toUpperCase(), ownerDocument: document,
      parentElement: parent, isConnected: true, visible: true,
      innerText: text, textContent: text, type: tag === 'button' ? 'button' : '',
      outerHTML: `<${tag} id="${id}">${text}</${tag}>`,
      getAttribute: key => attrs[key] ?? null,
      hasAttribute: key => key in attrs,
      setAttribute: (key, value) => { attrs[key] = value; },
      matches: () => false,
      getBoundingClientRect: () => ({ width: node.visible ? 100 : 0, height: 20 }),
      contains(other) {
        for (let n = other; n; n = n.parentElement) if (n === node) return true;
        return false;
      },
    };
    nodes.push(node);
    return node;
  }
  const body = element('body', 'body', 'BACKGROUND PRIVATE DIALOG TEXT');
  document.documentElement = body;
  document.body = body;
  const background = element('button', 'background', 'BACKGROUND PRIVATE', body);
  const dialog = element('dialog', 'dialog', 'DIALOG TEXT', body);
  const button = element('button', 'save', 'Save', dialog);
  const heading = element('h2', 'heading', 'Dialog heading', dialog);
  const link = element('a', 'help', 'Help', dialog, { href: '/help' });
  link.href = 'https://example.test/help';
  document.getElementById = id => nodes.find(n => n.getAttribute('id') === id) || null;
  document.getElementsByTagName = tag => nodes.filter(n => n.tagName.toLowerCase() === tag);
  document.querySelectorAll = selector => {
    if (selector === '[') throw new SyntaxError('invalid selector');
    if (selector === '*') return nodes.filter(n => n.isConnected);
    if (selector === ':modal' || selector === '[aria-modal="true"]') return [];
    if (selector.startsWith('a[href],button')) return nodes.filter(n => n.isConnected && ['BUTTON', 'A'].includes(n.tagName));
    if (selector === 'h1,h2,h3,h4') return [heading];
    if (selector === 'a[href]') return [link];
    if (selector === '#dialog button') return [button];
    if (selector === '#missing') return [];
    if (selector.startsWith('#')) return nodes.filter(n => n.isConnected && n.getAttribute('id') === selector.slice(1));
    return document.getElementsByTagName(selector);
  };
  document.evaluate = expression => {
    assert.equal(expression, '//dialog');
    let first = true;
    return { iterateNext: () => first ? (first = false, dialog) : null };
  };
  const window = {};
  const world = vm.createContext({ document, window, location: { href: 'https://example.test/' },
    getComputedStyle: n => ({ display: n.visible ? 'block' : 'none', visibility: 'visible' }) });
  let first = 1;
  function observe(query, html = false) {
    let code = script;
    for (const [marker, value] of Object.entries(substitutions)) code = code.replaceAll(marker, value);
    code = code.replaceAll('__GREPPY_REF_FIRST__', String(first))
      .replaceAll('__GREPPY_REF_LAST__', String(first + 199))
      .replaceAll('__GREPPY_SNAPSHOT__', JSON.stringify('doc-one'))
      .replaceAll('__GREPPY_INCLUDE_HTML__', String(html))
      .replaceAll('__GREPPY_QUERY__', JSON.stringify(query));
    first += 200;
    return JSON.parse(vm.runInContext(code, world));
  }
  return { observe, element, window, document, background, dialog, button, heading, link };
}

test('actual observation scopes text, headings, links and refs before serialization', () => {
  const f = fixture();
  const tree = f.observe('role=dialog', true);
  assert.equal(tree.text, 'DIALOG TEXT');
  assert.deepEqual(tree.headings, ['Dialog heading']);
  assert.deepEqual(tree.actionables.map(n => n.name), ['Save', 'Help']);
  assert.equal(tree.links.length, 1);
  assert.equal(tree.observation_scope.roots_returned, 1);
  assert.equal(tree.working_scope, null);
  assert.ok(!JSON.stringify(tree).includes('BACKGROUND PRIVATE'));
  assert.ok(tree.scoped_html.startsWith('<dialog'));
  const reference = Number(tree.actionables[0].ref.slice(1));
  assert.equal(f.window.__greppyObservedRefs.matches(f.button, reference), true);
  assert.equal(f.window.__greppyObservedRefs.matches(f.background, reference), false);
});

test('closed and absent regions produce empty data, never document fallback', () => {
  const f = fixture();
  f.dialog.visible = false;
  for (const query of ['role=dialog', 'css=#missing']) {
    const tree = f.observe(query, true);
    assert.equal(tree.observation_scope.roots_returned, 0);
    assert.equal(tree.text, '');
    assert.equal(tree.scoped_html, '');
    assert.deepEqual(tree.actionables, []);
    assert.deepEqual(tree.links, []);
    assert.deepEqual(tree.headings, []);
  }
});

test('shared query resolver supports CSS, XPath, id, tag, text and implicit role', () => {
  for (const query of ['css=#dialog', '#dialog', 'xpath=//dialog', 'id=dialog',
    'tag=dialog', 'text=DIALOG TEXT', 'text~/^DIALOG TEXT$/', 'role=dialog']) {
    const tree = fixture().observe(query);
    assert.equal(tree.text, 'DIALOG TEXT', query);
    assert.equal(tree.observation_scope.query, query);
  }
  assert.deepEqual(fixture().observe('css=#dialog button').actionables.map(n => n.name), ['Save']);
});

test('invalid query cannot replace the registry; next observation retains node refs', () => {
  const f = fixture();
  const initial = f.observe('role=dialog');
  const registry = f.window.__greppyObservedRefs;
  for (const query of ['css=[', 'unknown=dialog', 'role~dialog', 'text~/[/']) {
    assert.throws(() => f.observe(query));
    assert.equal(f.window.__greppyObservedRefs, registry);
  }
  const after = f.observe('role=dialog');
  assert.equal(after.actionables[0].ref, initial.actionables[0].ref);
  assert.equal(after.ref_snapshot, initial.ref_snapshot);
});

test('no query preserves unfiltered content; HTML is opt-in', () => {
  const tree = fixture().observe(null);
  assert.ok(tree.text.includes('BACKGROUND PRIVATE'));
  assert.equal(tree.actionables.length, 3);
  assert.equal(tree.observation_scope, undefined);
  assert.equal(tree.scoped_html, undefined);
});

test('a replaced node never inherits the old ref even with the same selector', () => {
  const f = fixture();
  const first = f.observe('role=dialog');
  const oldRef = Number(first.actionables[0].ref.slice(1));
  f.button.isConnected = false;
  assert.equal(f.window.__greppyObservedRefs.matches(f.button, oldRef), false);
  const next = f.observe('role=dialog');
  assert.ok(!next.actionables.some(n => n.ref === '@' + oldRef));
});

test('actual script caps actionables only after scoping and reports text truncation', () => {
  const f = fixture();
  f.dialog.innerText = 'x'.repeat(8001);
  for (let i = 0; i < 205; i++) f.element('button', 'inside-' + i, 'Inside ' + i, f.dialog);
  const tree = f.observe('role=dialog');
  assert.equal(tree.actionables.length, 200);
  assert.equal(tree.ref_count, 200);
  assert.equal(tree.refs_truncated, true);
  assert.equal(tree.text.length, 8000);
  assert.equal(tree.observation_scope.text_truncated, true);
  assert.ok(!tree.actionables.some(n => n.name === 'BACKGROUND PRIVATE'));
});

test('scope query data containing template marker text stays literal', () => {
  const query = 'text=__GREPPY_INCLUDE_HTML____GREPPY_REF_LIMIT__';
  const tree = fixture().observe(query);
  assert.equal(tree.observation_scope.query, query);
  assert.equal(tree.observation_scope.roots_returned, 0);
});

test('hidden descendants do not enter scoped heading/link projections', () => {
  const f = fixture();
  f.heading.visible = false;
  f.link.visible = false;
  const tree = f.observe('role=dialog');
  assert.deepEqual(tree.headings, []);
  assert.deepEqual(tree.links, []);
  assert.deepEqual(tree.actionables.map(n => n.name), ['Save']);
});
