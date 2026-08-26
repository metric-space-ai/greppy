import { chromium } from "playwright";

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
const popupPromise = page.waitForEvent("popup");
await page.evaluate(() => window.open("about:blank"));
const popup = await popupPromise;
if (!popup) {
  throw new Error("expected a popup page");
}
const opener = await popup.opener();
if (!opener) {
  throw new Error("popup opener was null");
}
if (opener !== page && opener._id !== page._id) {
  throw new Error("opener is not the creating page");
}
if ((await page.opener()) !== null) {
  throw new Error("top-level page opener must be null");
}
await browser.close();
