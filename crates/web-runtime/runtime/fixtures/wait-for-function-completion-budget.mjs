import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!doctype html><title>Public completion budget</title><body>ready</body>");

for (const expected of [true, { answer: 42, nested: { ok: true } }]) {
  await page.evaluate(() => { window.completionPredicateCalls = 0; });
  const actual = await page.waitForFunction((value) => {
    window.completionPredicateCalls++;
    const key = Object.keys(window).filter(k => k.indexOf("__greppyWait_") === 0).pop();
    const slot = window[key];
    let done = 0;
    // Delay only the destructive completion read, after this predicate returns.
    Object.defineProperty(slot, "done", {
      configurable: true,
      get() {
        const end = Date.now() + 120;
        while (Date.now() < end) {}
        return done;
      },
      set(value) { done = value; },
    });
    return value;
  }, expected, { timeout: 2000 });
  if ((typeof expected === "boolean" ? actual !== expected : actual?.answer !== expected.answer || actual?.nested?.ok !== expected.nested.ok || Object.keys(actual).length !== 2 || Object.keys(actual.nested).length !== 1)) {
    throw new Error("waitForFunction lost completed value: " + JSON.stringify(actual));
  }
  const state = await page.evaluate(() => ({
    calls: window.completionPredicateCalls,
    slots: Object.keys(window).filter(k => k.indexOf("__greppyWait_") === 0).length,
  }));
  if (state.calls !== 1 || state.slots !== 0) {
    throw new Error("completion was replayed or retained: " + JSON.stringify(state));
  }
}
await browser.close();
