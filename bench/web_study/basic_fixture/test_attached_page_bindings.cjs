'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const root = path.resolve(__dirname, '../../..');
const jsRoot = path.join(root, 'crates/web-runtime/runtime/js');
const helper = fs.readFileSync(path.join(jsRoot, 'attached-page-bindings.js'), 'utf8');
const bind = vm.runInNewContext(helper + '\ngreppyAttachedPageBindings;');
function descriptor() {
  return { schema: 'greppy.web.attached-page.v1', selected_page: 'page-b',
    browser: { id: 'browser-a', generation: 4 }, contexts: [
      { id: 'context-a', generation: 7, pages: [
        { id: 'page-a', generation: 11, url: 'http://fixture/a' },
        { id: 'page-b', generation: 12, url: 'http://fixture/b' }] },
      { id: 'context-b', generation: 9, pages: [
        { id: 'page-c', generation: 13, url: 'http://fixture/c' }] }] };
}
function factories(log) {
  class Browser { constructor(id, generation) { log.push(id); Object.assign(this, { _id: id, _generation: generation, _contexts: [] }); } }
  class BrowserContext { constructor(id, generation) { log.push(id); Object.assign(this, { _id: id, _generation: generation, _pages: [] }); } }
  class Page { constructor(id, generation) { log.push(id); Object.assign(this, { _id: id, _generation: generation }); } }
  return { Browser, BrowserContext, Page };
}

test('restores the exact graph including inactive siblings and generations', () => {
  const log = [];
  const result = bind(descriptor(), 'page-b', factories(log));
  assert.equal(result.borrowed, true);
  assert.equal(result.page._id, 'page-b');
  assert.equal(result.page._generation, 12);
  assert.equal(result.context._generation, 7);
  assert.equal(result.browser._generation, 4);
  assert.equal(result.page._url, 'http://fixture/b');
  assert.equal(result.page._context, result.context);
  assert.equal(result.context._browser, result.browser);
  assert.equal(result.context._lastPage, 'page-b');
  assert.equal(result.browser._contexts[1]._pages[0]._id, 'page-c');
  assert.equal(log.length, 6);
});

test('rejects invalid identity graphs before constructing any object', () => {
  const mutations = [
    d => { d.schema = 'greppy.web.attached-page.v2'; },
    d => { d.selected_page = 'page-foreign'; },
    d => { d.contexts[0].pages.pop(); },
    d => { d.contexts[1].pages[0].id = 'page-b'; },
    d => { d.contexts[1].id = 'context-a'; },
    d => { d.contexts[0].pages[0].id = 'browser-a'; },
    d => { d.contexts[0].generation = 0; },
    d => { d.contexts[0].pages[1].generation = 1.5; },
    d => { d.browser.generation = Number.MAX_SAFE_INTEGER + 1; },
    d => { d.contexts[0].pages[1].url = null; },
  ];
  for (const mutate of mutations) {
    const d = descriptor(); mutate(d); const log = [];
    assert.throws(() => bind(d, 'page-b', factories(log)), error => error.code === 'invalid_attached_page');
    assert.equal(log.length, 0);
  }
});

test('never falls back to the first page for absent or foreign targets', () => {
  for (const page of ['', undefined, 'page-foreign']) {
    const log = [];
    assert.throws(() => bind(descriptor(), page, factories(log)), error => error.code === 'invalid_attached_page');
    assert.equal(log.length, 0);
  }
});

test('does not mutate the supplied descriptor or use page text as metadata', () => {
  const d = descriptor(); d.contexts[0].pages[1].text = 'session was not found';
  const original = JSON.stringify(d);
  const result = bind(d, 'page-b', factories([]));
  assert.equal(JSON.stringify(d), original);
  assert.equal(result.page._id, 'page-b');
});

test('real facade classes attach without engine calls and target the existing page', async () => {
  const calls = [];
  const testDeno = { core: { ops: { op_engine_call(method, params) {
    calls.push({ method, params });
    if (method === 'page.title') return { title: 'Existing state' };
    if (method === 'locator.focus' || method === 'page.keyboard.press') return {};
    throw new Error('unexpected engine call: ' + method);
  } } } };
  {
    const filename = path.join(jsRoot, 'playwright.mjs');
    const source = fs.readFileSync(filename, 'utf8')
      .replace(/^export \{[^\n]*\};$/gm, '')
      .replace(/^export default [^\n]*;$/gm, '')
      .replace(/^export const /gm, 'const ');
    const facade = vm.runInNewContext(source + '\n({ Browser, BrowserContext, Page });',
      { Deno: testDeno }, { filename });
    const result = bind(descriptor(), 'page-b', facade);
    assert.equal(calls.length, 0);
    assert.equal(result.page.context(), result.context);
    assert.equal(result.context.browser(), result.browser);
    assert.equal(result.context.pages()[1], result.page);
    assert.equal(await result.page.title(), 'Existing state');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].method, 'page.title');
    assert.equal(calls[0].params.page, 'page-b');
    await result.page.locator('#field').press('Enter');
    assert.equal(calls.length, 3);
    assert.equal(calls[1].method, 'locator.focus');
    assert.equal(calls[1].params.page, 'page-b');
    assert.equal(calls[1].params.generation, 12);
    assert.equal(calls[2].method, 'page.keyboard.press');
    assert.equal(calls[2].params.page, 'page-b');
    assert.equal(calls[2].params.key, 'Enter');
    assert.equal(result.browser.isConnected(), true);
    assert.equal(result.page._closed, false);
  }
});
