import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
const value = await page.evaluate((n) => n + 1, 41);
if (value !== 42) {
  throw new Error("evaluate arg expected 42, got " + JSON.stringify(value));
}
await browser.close();
