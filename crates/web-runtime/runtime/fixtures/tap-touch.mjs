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

async function expectTouch(label) {
  const seq = await page.evaluate(() => window.__seq.slice());
  if (!seq.some((item) => String(item).startsWith("touchstart"))) {
    throw new Error(label + " produced no touchstart: " + JSON.stringify(seq));
  }
  if (!seq.includes("touchend")) {
    throw new Error(label + " produced no touchend: " + JSON.stringify(seq));
  }
  await page.evaluate(() => {
    window.__seq = [];
  });
}

await page.locator("#t").tap();
await expectTouch("locator.tap");
await page.mainFrame().tap("#t");
await expectTouch("frame.tap");
await page.tap("#t");
await expectTouch("page.tap");
const box = await page.locator("#t").boundingBox();
await page.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
await expectTouch("touchscreen.tap");

await page.setContent(`<!DOCTYPE html><html><body>
<iframe name="child" srcdoc="<button id='t' style='width:80px;height:40px'>Tap</button>"></iframe>
</body></html>`);
const child = await page.frame({ name: "child" });
if (!child) throw new Error("missing child frame for tap");
await child.waitForSelector("#t");
await child.evaluate(() => {
  window.__seq = [];
  const el = document.getElementById("t");
  el.addEventListener("touchstart", (event) => {
    window.__seq.push("touchstart:" + ((event.touches && event.touches.length) || 0));
  });
  el.addEventListener("touchend", () => {
    window.__seq.push("touchend");
  });
});
await child.tap("#t");
const childSeq = await child.evaluate(() => window.__seq.slice());
if (!childSeq.some((item) => String(item).startsWith("touchstart"))) {
  throw new Error("child Frame.tap produced no touchstart: " + JSON.stringify(childSeq));
}
if (!childSeq.includes("touchend")) {
  throw new Error("child Frame.tap produced no touchend: " + JSON.stringify(childSeq));
}
await browser.close();
