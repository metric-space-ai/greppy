import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  "<!DOCTYPE html><html><body><input id='q'><p id='out'></p></body></html>",
);
await page.locator("#q").fill("ok");
const filled = await page.evaluate(() => document.querySelector("#q").value);
if (filled !== "ok") {
  throw new Error("fill expected ok, got " + JSON.stringify(filled));
}
await browser.close();
