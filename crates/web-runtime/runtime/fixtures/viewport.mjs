import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><p id='v'>vp</p></body></html>");

const before = await page.viewportSize();
if (!before || before.width !== 1280 || before.height !== 720) {
  throw new Error("viewportSize must report the Playwright default 1280x720, got " + JSON.stringify(before));
}
const innerBefore = await page.evaluate(() => ({
  width: window.innerWidth,
  height: window.innerHeight,
}));
if (innerBefore.width !== 1280 || innerBefore.height !== 720) {
  throw new Error("window inner size must match the default viewport, got " + JSON.stringify(innerBefore));
}

await page.setViewportSize({ width: 1024, height: 768 });
const after = await page.viewportSize();
if (!after || after.width !== 1024 || after.height !== 768) {
  throw new Error("setViewportSize did not apply: " + JSON.stringify({ before, after }));
}
const innerAfter = await page.evaluate(() => ({
  width: window.innerWidth,
  height: window.innerHeight,
}));
if (innerAfter.width !== 1024 || innerAfter.height !== 768) {
  throw new Error("window inner size did not follow setViewportSize: " + JSON.stringify({ innerBefore, innerAfter }));
}

await browser.close();
