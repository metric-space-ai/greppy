import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<button id="t" style="width:80px;height:40px">Tap</button>
</body></html>`);
await page.evaluate(() => {
  window.__seq = [];
  const el = document.getElementById("t");
  el.addEventListener("touchstart", (event) => {
    window.__seq.push("touchstart:" + ((event.touches && event.touches.length) || 0));
  });
  el.addEventListener("touchend", () => {
    window.__seq.push("touchend");
  });
  el.addEventListener("mousedown", () => {
    window.__seq.push("mousedown");
  });
});
await page.locator("#t").tap();
const afterLocator = await page.evaluate(() => window.__seq.slice());
if (!afterLocator.some((item) => String(item).startsWith("touchstart"))) {
  throw new Error("locator.tap produced no touchstart: " + JSON.stringify(afterLocator));
}
if (!afterLocator.includes("touchend")) {
  throw new Error("locator.tap produced no touchend: " + JSON.stringify(afterLocator));
}
await page.evaluate(() => {
  window.__seq = [];
});
await page.tap("#t");
const afterPage = await page.evaluate(() => window.__seq.slice());
if (!afterPage.some((item) => String(item).startsWith("touchstart"))) {
  throw new Error("page.tap produced no touchstart: " + JSON.stringify(afterPage));
}
const box = await page.locator("#t").boundingBox();
await page.evaluate(() => {
  window.__seq = [];
});
await page.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
const afterScreen = await page.evaluate(() => window.__seq.slice());
if (!afterScreen.some((item) => String(item).startsWith("touchstart"))) {
  throw new Error("touchscreen.tap produced no touchstart: " + JSON.stringify(afterScreen));
}
await browser.close();
