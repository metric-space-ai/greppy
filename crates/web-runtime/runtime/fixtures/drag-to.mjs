import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<div id="src" style="width:80px;height:40px;background:#fcc">src</div>
<div id="dst" style="width:80px;height:40px;background:#cfc;margin-top:40px">dst</div>
</body></html>`);
await page.evaluate(() => {
  window.__drag = [];
  document.getElementById("src").addEventListener("mousedown", () => window.__drag.push("src-down"));
  document.getElementById("dst").addEventListener("mouseup", () => window.__drag.push("dst-up"));
});
await page.locator("#src").dragTo(page.locator("#dst"), { timeout: 5_000 });
const seq = await page.evaluate(() => window.__drag);
if (!seq.includes("src-down") || !seq.includes("dst-up")) {
  throw new Error("dragTo mouse path missing: " + JSON.stringify(seq));
}
await page.evaluate(() => {
  window.__drag = [];
});
await page.dragAndDrop("#src", "#dst");
const seq2 = await page.evaluate(() => window.__drag);
if (!seq2.includes("src-down") || !seq2.includes("dst-up")) {
  throw new Error("dragAndDrop mouse path missing: " + JSON.stringify(seq2));
}
await browser.close();
