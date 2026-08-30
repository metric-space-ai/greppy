import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><div id='app'>boot</div><div id='a'>x</div><div id='b'>y</div></body></html>");
await page.evaluate(() => {
  window.__origSetTimeout = window.setTimeout;
  window.__waitFinishCount = 0;
});

const started = Date.now();
await page.waitForFunction(() => {
  window.__predicateCalls = (window.__predicateCalls || 0) + 1;
  if (!window.__hydrateArmed) {
    window.__hydrateArmed = true;
    window.__hydrateTicks = 0;
    setInterval(function () {
      window.__hydrateTicks += 1;
    }, 16);
    setTimeout(function () {
      document.getElementById("app").textContent = "hydrated";
    }, 80);
    return false;
  }
  return document.getElementById("app").textContent === "hydrated";
});
const elapsed = Date.now() - started;
if (elapsed < 50) {
  throw new Error("waitForFunction returned before the hydrate timer " + elapsed);
}
const text = (await page.locator("#app").innerText()).trim();
if (text !== "hydrated") throw new Error("spa hydrate " + text);
const ticks = await page.evaluate(() => window.__hydrateTicks);
if (ticks < 2) {
  throw new Error("Servo event loop did not run during waitForFunction: " + ticks);
}
const calls = await page.evaluate(() => window.__predicateCalls);
if (calls < 1) {
  throw new Error("predicate never ran");
}
const leftoverAfterSuccess = await page.evaluate(() =>
  Object.getOwnPropertyNames(window).filter((k) => k.indexOf("__greppyWait") === 0),
);
if (leftoverAfterSuccess.length) {
  throw new Error("leftover wait tokens after success: " + leftoverAfterSuccess.join(","));
}
const sameTimeout = await page.evaluate(() => window.setTimeout === window.__origSetTimeout);
if (!sameTimeout) {
  throw new Error("waitForFunction replaced window.setTimeout");
}
const finishesAfterSuccess = await page.evaluate(() => window.__waitFinishCount);
if (finishesAfterSuccess !== 1) {
  throw new Error("expected one waiter completion, got " + finishesAfterSuccess);
}
await page.waitForTimeout(120);
const callsAfterIdle = await page.evaluate(() => window.__predicateCalls);
if (callsAfterIdle !== calls) {
  throw new Error("residual waiter callback after success: " + calls + " -> " + callsAfterIdle);
}
const finishesAfterIdle = await page.evaluate(() => window.__waitFinishCount);
if (finishesAfterIdle !== 1) {
  throw new Error("second completion after success: " + finishesAfterIdle);
}

await page.evaluate(() => {
  setTimeout(function () {
    document.getElementById("a").textContent = "A";
  }, 40);
  setTimeout(function () {
    document.getElementById("b").textContent = "B";
  }, 70);
});
await page.waitForFunction(() => document.getElementById("a").textContent === "A");
await page.waitForFunction(() => document.getElementById("b").textContent === "B");
const dualTimeout = await page.evaluate(() => window.setTimeout === window.__origSetTimeout);
if (!dualTimeout) {
  throw new Error("parallel waiters replaced window.setTimeout");
}
const leftoverAfterParallel = await page.evaluate(() =>
  Object.getOwnPropertyNames(window).filter((k) => k.indexOf("__greppyWait") === 0),
);
if (leftoverAfterParallel.length) {
  throw new Error("leftover wait tokens after parallel waiters: " + leftoverAfterParallel.join(","));
}

let timeoutErr = null;
try {
  await page.waitForFunction(() => false, null, { timeout: 250 });
} catch (error) {
  timeoutErr = String(error && error.message ? error.message : error);
}
if (!timeoutErr || !/timeout/i.test(timeoutErr)) {
  throw new Error("expected waitForFunction timeout, got " + timeoutErr);
}
const leftoverAfterTimeout = await page.evaluate(() =>
  Object.getOwnPropertyNames(window).filter((k) => k.indexOf("__greppyWait") === 0),
);
if (leftoverAfterTimeout.length) {
  throw new Error("leftover wait tokens after timeout: " + leftoverAfterTimeout.join(","));
}
await page.waitForTimeout(80);
const leftoverAfterTimeoutIdle = await page.evaluate(() =>
  Object.getOwnPropertyNames(window).filter((k) => k.indexOf("__greppyWait") === 0),
);
if (leftoverAfterTimeoutIdle.length) {
  throw new Error("residual wait tokens after timeout: " + leftoverAfterTimeoutIdle.join(","));
}
await browser.close();
