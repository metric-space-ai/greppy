// Controller-module contract only. No browser, page, engine or network is started.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import vm from 'node:vm';

const [entryPath, wrapperPath] = process.argv.slice(2);
const calls = [];
const boundary = new Error('native_engine_boundary_not_executed');
const context = vm.createContext({
  fixtureUrl: 'http://127.0.0.1:1/',
  Deno: { core: { ops: {
    op_trace_limit_bytes: () => 8 * 1024 * 1024,
    op_engine_call(method, params) {
      calls.push({ method, params: JSON.parse(JSON.stringify(params)) });
      throw boundary;
    },
  } } },
});
const entry = new vm.SourceTextModule(readFileSync(entryPath, 'utf8'), { context });
const dependencies = [...entry.dependencySpecifiers];
assert.deepEqual(dependencies, ['playwright']);
const wrapperSource = readFileSync(wrapperPath);
const wrapper = new vm.SourceTextModule(wrapperSource.toString('utf8'), { context });
await entry.link((specifier) => {
  assert.equal(specifier, 'playwright');
  return wrapper;
});

// Execute the actual generated entry and real wrapper up to its first engine op.
// A mismatched launch option fails in the wrapper before this boundary.
await assert.rejects(entry.evaluate(), (error) => error === boundary);
assert.deepEqual(calls, [{ method: 'chromium.launch', params: { timeout: 30000 } }]);

// Do not permit the adapter to silently discard quality/isolation/viewport inputs.
const browserType = entry.namespace.nativeHeadlessBrowserType(wrapper.namespace.chromium);
for (const expression of [
  'undefined', 'null', '{}', '({headless:false})',
  '({headless:true,viewport:{width:1,height:1}})',
  '({headless:true,proxy:{server:"unused"}})',
  'Object.assign(Object.create({viewport:{width:1}}), {headless:true})',
  'Object.defineProperty({headless:true}, "viewport", {value:{width:1}})',
]) {
  await assert.rejects(browserType.launch(vm.runInContext(expression, context)),
    /native efficiency launch requires exactly/);
}
assert.equal(calls.length, 1);
console.log(JSON.stringify({ dependencies, launch_policy: 'native-fixed-headless-v1',
  wrapper_sha256: createHash('sha256').update(wrapperSource).digest('hex'),
  boundary: 'controller only; first engine operation deliberately refused' }));
process.exit(0);
