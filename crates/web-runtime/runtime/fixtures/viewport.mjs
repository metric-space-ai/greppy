import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><p id='v'>vp</p></body></html>");

const before = await page.viewportSize();
if (!before || before.width !== 800 || before.height !== 600) {
  throw new Error("viewportSize must report the real Servo size 800x600, got " + JSON.stringify(before));
}
const innerBefore = await page.evaluate(() => ({
  width: window.innerWidth,
  height: window.innerHeight,
}));

async function expectUnsupportedSetViewport(label, size) {
  let failed = false;
  try {
    await page.setViewportSize(size);
  } catch (error) {
    const message = String(error.message);
    if (!message.includes("unsupported_playwright_operation: Page.setViewportSize")) {
      throw new Error(label + " wrong error: " + message);
    }
    failed = true;
  }
  if (!failed) {
    throw new Error(label + " must fail closed as unsupported_playwright_operation: Page.setViewportSize");
  }
}

await expectUnsupportedSetViewport("setViewportSize 1024x768", { width: 1024, height: 768 });
await expectUnsupportedSetViewport("setViewportSize current 800x600", { width: 800, height: 600 });

const after = await page.viewportSize();
if (after.width !== before.width || after.height !== before.height) {
  throw new Error("viewportSize changed without a real resize: " + JSON.stringify({ before, after }));
}
const innerAfter = await page.evaluate(() => ({
  width: window.innerWidth,
  height: window.innerHeight,
}));
if (innerAfter.width !== innerBefore.width || innerAfter.height !== innerBefore.height) {
  throw new Error("window inner size changed without a real resize: " + JSON.stringify({ innerBefore, innerAfter }));
}

await browser.close();
