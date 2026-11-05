const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../js/navigation-document-probe.js'), 'utf8');
const selector = '*:not(html):not(head):not(body)';
function probe(document) {
  return vm.runInNewContext(source, { document });
}

test('a populated document and a body-less frameset need no layout or HTML copy', () => {
  for (const tagName of ['P', 'FRAMESET']) {
    let queries = 0;
    const document = {
      querySelector(query) { assert.equal(query, selector); queries++; return { tagName }; },
      querySelectorAll() { throw new Error('must not enumerate the document'); },
      get body() { throw new Error('must not read layout text'); },
      get documentElement() { throw new Error('must not serialize HTML'); },
    };
    assert.equal(probe(document), false);
    assert.equal(queries, 1);
  }
});

test('shell-only body text proves content without HTML serialization', () => {
  assert.equal(probe({
    querySelector: () => null,
    body: { innerText: '   visible text   ' },
    get documentElement() { throw new Error('text already proves content'); },
  }), false);
});

test('only an empty, small shell is positively identified as missing', () => {
  const document = {
    querySelector: () => null,
    body: { innerText: ' \n\t' },
    documentElement: { outerHTML: '<html><head></head><body></body></html>' },
  };
  assert.equal(probe(document), true);
  document.documentElement.outerHTML = 'x'.repeat(500);
  assert.equal(probe(document), false);
});

test('unavailable evidence throws instead of claiming an empty document', () => {
  assert.throws(() => probe({
    querySelector: () => null,
    get body() { throw new Error('script thread unavailable'); },
  }), /script thread unavailable/);
});
