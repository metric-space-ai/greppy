import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
page.setDefaultTimeout(800);
await page.setContent(`<!DOCTYPE html><html><body>
<button id="go">Go</button>
<button id="off" disabled>Off</button>
<button id="hid" style="display:none">Hid</button>
<button id="invis" style="visibility:hidden">Invis</button>
<input id="ro" readonly value="no">
</body></html>`);
await page.locator("#go").click();
if (await page.locator("#hid").isVisible()) throw new Error("display:none isVisible");
if (await page.locator("#invis").isVisible()) throw new Error("visibility:hidden isVisible");

async function expectTimeout(label, fn) {
  const started = Date.now();
  try {
    await fn();
  } catch (error) {
    const message = String(error.message);
    if (!(message.includes("timed out") || message.includes("actionable"))) throw error;
    if (message.includes("html=") || message.includes("<html") || message.includes("outerHTML")) {
      throw new Error(label + " dumped page HTML: " + message);
    }
    if (Date.now() - started > 5_000) throw new Error(label + " ignored page timeout");
    return message;
  }
  throw new Error(label + " succeeded");
}

await expectTimeout("disabled click", () => page.locator("#off").click());
await expectTimeout("display:none click", () => page.locator("#hid").click());
await expectTimeout("visibility:hidden click", () => page.locator("#invis").click());
await expectTimeout("readonly fill", () => page.locator("#ro").fill("x"));

await page.setContent(`<!DOCTYPE html><html><body style="margin:0">
<button id="under">Covered</button>
<div id="mask" style="position:fixed;left:0;top:0;right:0;bottom:0;background:rgba(0,0,0,0.01)"></div>
</body></html>`);
await expectTimeout("overlay click", () => page.locator("#under").click());

page.setDefaultTimeout(3_000);
await page.setContent(`<!DOCTYPE html><html><body style="margin:0">
<button id="later">Later</button>
<div id="mask" style="position:fixed;left:0;top:0;right:0;bottom:0"></div>
</body></html>`);
await page.evaluate(() => {
  window.__later = 0;
  document.getElementById("later").addEventListener("click", () => {
    window.__later += 1;
  });
  setTimeout(() => {
    const mask = document.getElementById("mask");
    if (mask) mask.remove();
  }, 250);
});
await page.locator("#later").waitFor({ state: "visible" });
await page.getByRole("button", { name: "Later" }).click();
const later = await page.evaluate(() => window.__later);
if (later < 1) throw new Error("auto-wait overlay click " + later);
let hiddenState = false;
try {
  await page.locator("#later").waitFor({ state: "hidden" });
} catch (error) {
  hiddenState = String(error.message).includes("unsupported_playwright_operation");
}
if (!hiddenState) throw new Error("waitFor hidden must fail closed");

page.setDefaultTimeout(3_000);
await page.setContent(`<!DOCTYPE html><html><body>
<div style="height:2500px"></div>
<button id="low">Low</button>
</body></html>`);
await page.evaluate(() => {
  window.__low = 0;
  document.getElementById("low").addEventListener("click", () => {
    window.__low = 1;
  });
});
await page.locator("#low").click();
if ((await page.evaluate(() => window.__low)) !== 1) {
  throw new Error("off-screen click did not land");
}

page.setDefaultTimeout(1500);
await page.setContent(`<!DOCTYPE html><html><body style="margin:0;height:100vh">
<button id="jitter" style="position:absolute;left:8px;top:40px;width:80px;height:40px">Jitter</button>
</body></html>`);
await page.evaluate(() => {
  const el = document.getElementById("jitter");
  window.__jitterMoves = 0;
  window.__jitterClicks = 0;
  el.addEventListener("click", () => {
    window.__jitterClicks += 1;
  });
  let x = 8;
  window.__jitterTimer = setInterval(() => {
    x += 40;
    el.style.left = x + "px";
    window.__jitterMoves += 1;
  }, 16);
});
async function expectUnstableClick(label) {
  const unstable = await expectTimeout(label, () => page.locator("#jitter").click());
  if (!unstable.includes("failed_check=stable")) {
    throw new Error(label + " must name failed_check=stable: " + unstable);
  }
  return unstable;
}
await expectUnstableClick("unstable click 1");
await expectUnstableClick("unstable click 2");
await expectUnstableClick("unstable click 3");
const jitterState = await page.evaluate(() => ({
  moves: window.__jitterMoves,
  clicks: window.__jitterClicks,
  leaked: Object.getOwnPropertyNames(window).filter((key) => key.startsWith("__greppyPump")),
}));
if (jitterState.clicks !== 0) {
  throw new Error("moving element was clicked: " + JSON.stringify(jitterState));
}
if (jitterState.moves < 2) {
  throw new Error("Servo event loop did not run movement timers: " + JSON.stringify(jitterState));
}
if (jitterState.leaked.length !== 0) {
  throw new Error("pump tokens leftover after repeated timeout cycles: " + JSON.stringify(jitterState));
}
await page.evaluate(() => {
  if (window.__jitterTimer) clearInterval(window.__jitterTimer);
});

page.setDefaultTimeout(3_000);
await page.setContent(`<!DOCTYPE html><html><body style="margin:0;height:100vh">
<button id="settle" style="position:absolute;left:8px;top:40px;width:80px;height:40px">Settle</button>
</body></html>`);
await page.evaluate(() => {
  const el = document.getElementById("settle");
  window.__settle = 0;
  el.addEventListener("click", () => {
    window.__settle = 1;
  });
  let x = 8;
  window.__settleTimer = setInterval(() => {
    x += 40;
    el.style.left = x + "px";
  }, 16);
  setTimeout(() => {
    if (window.__settleTimer) clearInterval(window.__settleTimer);
    el.style.left = "40px";
  }, 220);
});
await page.locator("#settle").click();
if ((await page.evaluate(() => window.__settle)) !== 1) {
  throw new Error("click must wait until layout is stable");
}
const leakedAfterSettle = await page.evaluate(() =>
  Object.getOwnPropertyNames(window).filter((key) => key.startsWith("__greppyPump"))
);
if (leakedAfterSettle.length !== 0) {
  throw new Error("pump tokens leftover after successful settle click: " + JSON.stringify(leakedAfterSettle));
}

await browser.close();
