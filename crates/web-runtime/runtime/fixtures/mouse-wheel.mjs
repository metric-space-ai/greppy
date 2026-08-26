import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<div id="scroller" style="width:120px;height:80px;overflow:auto">
  <div style="height:400px">wheel-me</div>
</div>
</body></html>`);
await page.evaluate(() => {
  window.__wheel = [];
  document.getElementById("scroller").addEventListener("wheel", (event) => {
    window.__wheel.push({ x: event.deltaX, y: event.deltaY });
  });
});
const box = await page.locator("#scroller").boundingBox();
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
await page.mouse.wheel(0, 80);
const wheel = await page.evaluate(() => window.__wheel);
if (!wheel.length || !wheel.some((item) => Number(item.y) !== 0)) {
  throw new Error("mouse.wheel produced no wheel deltaY: " + JSON.stringify(wheel));
}
await browser.close();
