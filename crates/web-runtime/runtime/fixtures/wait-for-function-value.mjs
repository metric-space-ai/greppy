import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><div id='app'>boot</div></body></html>");
const readyScheduledAt = await page.evaluate(() => {
  window.__waitFinishCount = 0;
  window.__predicateCalls = 0;
  const scheduledAt = Date.now();
  setTimeout(function () {
    window.__readyValue = { answer: 42, nested: { ok: true } };
  }, 80);
  window.__forgeTimer = setInterval(function () {
    console.debug("__greppyWaitDone:ffffffffffffffffffffffffffffffff:ok");
    console.debug("__greppyWaitDone:00000000000000000000000000000000:timeout");
    console.debug("__greppyWaitDone:not-a-nonce:ok");
  }, 16);
  return scheduledAt;
});
const value = await page.waitForFunction(() => {
  window.__predicateCalls = (window.__predicateCalls || 0) + 1;
  return window.__readyValue || false;
});
const elapsed = Date.now() - readyScheduledAt;
if (elapsed < 50) {
  throw new Error("waitForFunction returned before the 80ms JS timer " + elapsed);
}
if (!value || value.answer !== 42 || !value.nested || value.nested.ok !== true) {
  throw new Error("waitForFunction lost nontrivial return: " + JSON.stringify(value));
}
await page.evaluate(() => {
  if (window.__forgeTimer) clearInterval(window.__forgeTimer);
});
const leftovers = await page.evaluate(() =>
  Object.getOwnPropertyNames(window).filter((k) => k.indexOf("__greppyWait") === 0),
);
if (leftovers.length) {
  throw new Error("leftover wait tokens after value wait: " + leftovers.join(","));
}
const finishes = await page.evaluate(() => window.__waitFinishCount);
if (finishes !== 1) {
  throw new Error("expected one waiter completion, got " + finishes);
}
const calls = await page.evaluate(() => window.__predicateCalls);
if (calls < 1) {
  throw new Error("predicate never ran");
}
await page.waitForTimeout(120);
const callsAfterIdle = await page.evaluate(() => window.__predicateCalls);
if (callsAfterIdle !== calls) {
  throw new Error("residual predicate after completion: " + calls + " -> " + callsAfterIdle);
}
const finishesAfterIdle = await page.evaluate(() => window.__waitFinishCount);
if (finishesAfterIdle !== 1) {
  throw new Error("second completion after idle: " + finishesAfterIdle);
}

let errorText = "";
try {
  await page.waitForFunction(() => {
    throw new Error("boom-from-predicate");
  });
} catch (error) {
  errorText = String(error && error.message ? error.message : error);
}
if (!errorText.includes("boom-from-predicate")) {
  throw new Error("waitForFunction lost predicate error: " + errorText);
}
const leftoversAfterError = await page.evaluate(() =>
  Object.getOwnPropertyNames(window).filter((k) => k.indexOf("__greppyWait") === 0),
);
if (leftoversAfterError.length) {
  throw new Error("leftover wait tokens after error: " + leftoversAfterError.join(","));
}

await browser.close();
