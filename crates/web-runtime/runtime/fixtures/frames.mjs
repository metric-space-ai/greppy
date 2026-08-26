import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  `<!DOCTYPE html><html><body><iframe name="child" srcdoc="<p id='in'>frame-ok</p>"></iframe></body></html>`,
);
const frames = await page.frames();
if (frames.length < 1) {
  throw new Error("expected iframe, got " + frames.length);
}
const child = await page.frame({ name: "child" });
if (!child) {
  throw new Error("missing named frame");
}
await browser.close();
