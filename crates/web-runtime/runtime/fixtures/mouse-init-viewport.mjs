import { chromium, devices } from "playwright";

const browser = await chromium.launch();
if (!String(browser.version()).includes("Servo")) {
  throw new Error("version " + browser.version());
}
let devicesClosed = false;
try {
  await devices.iPhone();
} catch (error) {
  devicesClosed = String(error.message).includes("unsupported_playwright_operation");
}
if (!devicesClosed) throw new Error("devices must fail closed");
const context = await browser.newContext();
if (!browser.isConnected()) throw new Error("isConnected");
if (browser.browserType().name() !== "chromium") {
  throw new Error("browserType " + browser.browserType().name());
}
if (!browser.contexts().some((item) => item === context)) {
  throw new Error("contexts missing current context");
}
context.setDefaultTimeout(5_000);
await context.addInitScript(() => {
  window.__ctxinit = 1;
});
const page = await context.newPage();
if (context.pages().length !== 1) throw new Error("pages");
if (page.context() !== context) throw new Error("context");
page.setDefaultTimeout(5_000);
const vp = await page.viewportSize();
if (vp.width !== 800 || vp.height !== 600) throw new Error("viewport " + JSON.stringify(vp));
await page.addInitScript(() => {
  window.__init = 1;
});
await page.setContent("<!DOCTYPE html><html><body><li>a</li><li>b</li><button id='m'>M</button></body></html>");
const init = await page.evaluate(() => window.__init);
if (init !== 1) throw new Error("init script " + JSON.stringify(init));
const ctxinit = await page.evaluate(() => window.__ctxinit);
if (ctxinit !== 1) throw new Error("context init script " + JSON.stringify(ctxinit));
const last = await page.locator("li").last().innerText();
if (last.trim() !== "b") throw new Error("last " + last);
const all = await page.locator("li").all();
if (all.length !== 2) throw new Error("all " + all.length);
await page.evaluate(() => {
  window.__clicks = 0;
  window.__downs = [];
  document.addEventListener("click", () => {
    window.__clicks += 1;
  });
  document.addEventListener("mousedown", (event) => {
    window.__downs.push({
      id: event.target && event.target.id,
      x: event.clientX,
      y: event.clientY,
    });
  });
});
const box = await page.locator("#m").boundingBox();
if (!box) throw new Error("missing #m box");
const mx = box.x + box.width / 2;
const my = box.y + box.height / 2;
await page.mouse.move(mx, my);
await page.mouse.down();
await page.mouse.up();
const downs = await page.evaluate(() => window.__downs);
if (!downs.some((down) => down.id === "m")) {
  throw new Error("mouse.down should hit last move target #m: " + JSON.stringify(downs));
}
await page.mouse.click(12, 12);
await page.mouse.dblclick(12, 12);
const clicks = await page.evaluate(() => window.__clicks);
if (clicks < 2) throw new Error("mouse.dblclick clicks " + clicks);
const main = page.mainFrame();
if ((await main.locator("li").count()) !== 2) throw new Error("main frame count");
if ((await main.getByText("a").count()) < 1) throw new Error("main getByText");
await browser.close();
