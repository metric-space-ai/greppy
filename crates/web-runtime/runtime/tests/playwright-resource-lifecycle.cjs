// Contract tests execute the shipped wrapper; the seam is its native IPC boundary.
// Native lifecycle fixture separately verifies actual Servo object ownership.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");
function fixture() {
  let sequence = 0;
  const calls = [];
  const records = { requests: [], responses: [] };
  const ops = {
    op_trace_limit_bytes: () => 8 * 1024 * 1024,
    op_engine_call: async (method, params) => {
      calls.push({ method, params });
      for (const key of ["page", "context", "browser"]) {
        if (params[key] != null) assert.equal(typeof params[key], "string", "owner leaked into wire payload");
      }
      if (method === "chromium.launch") return { browser: "browser-" + ++sequence };
      if (method === "browser.newContext") return { context: "context-" + ++sequence };
      if (method === "context.newPage") return { page: "page-" + ++sequence };
      if (method === "page.networkBaseline") return { seen: records.baseline || [] };
      if (method === "page.requests") return { requests: records.requests };
      if (method === "page.responses") return { responses: records.responses };
      if (method === "page.frames") return { frames: [] };
      if (method === "page.popups") return { popups: [] };
      if (method === "page.url") return { url: "https://fixture.invalid/" };
      if (method === "page.goto") return { url: params.url, status: 200 };
      if (["page.close", "context.close", "browser.close"].includes(method)) return {};
      throw new Error("unexpected engine call " + method);
    },
  };
  const world = vm.createContext({ Deno: { core: { ops } }, console });
  let source = fs.readFileSync(path.join(__dirname, "../js/playwright.mjs"), "utf8");
  source = source.replace(/^export \{.*\};$/gm, "").replace(/^export default .*;$/gm, "").replace(/^export const /gm, "const ");
  vm.runInContext(source, world, { filename: "shipped-playwright.mjs" });
  return { world, calls, records, chromium: vm.runInContext("chromium", world) };
}
test("closed pages and contexts release wrapper ownership while stale handles reject", async () => {
  const f = fixture();
  const browser = await f.chromium.launch();
  const context = await browser.newContext();
  let last;
  for (let i = 0; i < 500; i++) {
    const page = await context.newPage();
    const locator = page.locator("p");
    await page.close();
    assert.equal(context.pages().length, 0);
    assert.equal(vm.runInContext("pageContexts.size", f.world), 0);
    const before = f.calls.length;
    await assert.rejects(() => locator.textContent(), { code: "object_disposed" });
    await assert.rejects(() => page.title(), { code: "object_disposed" });
    assert.equal(f.calls.length, before);
    last = page;
  }
  assert.equal(last._emittedNetwork.size, 0);
  await context.close();
  assert.equal(browser.contexts().length, 0);
  await assert.rejects(() => context.newPage(), { code: "object_disposed" });
  await browser.close();
  await assert.rejects(() => browser.newContext(), { code: "object_disposed" });
});
test("close callback failures still release every page and native context", async () => {
  const f = fixture();
  const browser = await f.chromium.launch();
  const context = await browser.newContext();
  const first = await context.newPage();
  const second = await context.newPage();
  first.on("close", () => { throw new Error("callback failed"); });
  await assert.rejects(() => browser.close(), /callback failed/);
  assert.equal(first._closed, true);
  assert.equal(second._closed, true);
  assert.equal(context.pages().length, 0);
  assert.equal(vm.runInContext("pageContexts.size", f.world), 0);
  assert(f.calls.some(call => call.method === "context.close"));
  assert(f.calls.some(call => call.method === "browser.close"));
});
test("network dispatch joins stable request IDs once and bounds deduplication to retained history", async () => {
  const f = fixture();
  const browser = await f.chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  const seen = [];
  context.on("response", response => seen.push([response.request().method(), response.status()]));
  for (let i = 0; i < 200; i++) {
    f.records.requests = [{ requestId: String(i), url: "https://fixture.invalid/same", method: "POST" }];
    f.records.responses = [{ requestId: String(i), status: 201 }];
    await page._dispatchNetwork(true);
    await page._dispatchNetwork(true);
    assert(page._emittedNetwork.size <= 2);
  }
  assert.equal(seen.length, 200);
  assert(seen.every(([method, status]) => method === "POST" && status === 201));
  await browser.close();
});
test("navigation without network observers avoids historical snapshots and the implicit request wait", async () => {
  const f = fixture();
  const browser = await f.chromium.launch();
  const page = await browser.newPage();
  await page.goto("https://fixture.invalid/");
  assert(!f.calls.some(call => ["page.requests", "page.responses", "page.waitForRequest"].includes(call.method)));
  await browser.close();
});

test("a late network observer does not replay requests completed before subscription", async () => {
  const f = fixture();
  const browser = await f.chromium.launch();
  const page = await browser.newPage();
  await page.goto("https://fixture.invalid/old");
  f.records.baseline = [{ id: "old", failed: false, finished: true }];
  f.records.requests = [{ requestId: "old", url: "https://fixture.invalid/old" }];
  f.records.responses = [{ requestId: "old", url: "https://fixture.invalid/old", status: 200 }];
  const seen = [];
  page.on("response", response => seen.push(response.url()));
  f.records.requests.push({ requestId: "new", url: "https://fixture.invalid/new" });
  f.records.responses.push({ requestId: "new", url: "https://fixture.invalid/new", status: 201 });
  await page._dispatchNetwork(true);
  assert.deepEqual(seen, ["https://fixture.invalid/new"]);
  assert.equal(f.calls.filter(call => call.method === "page.networkBaseline").length, 1);
  await browser.close();
});
