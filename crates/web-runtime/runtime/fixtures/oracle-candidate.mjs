import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  "<!DOCTYPE html><html><head><title>Oracle</title></head><body><p id='x'>ok</p></body></html>",
);
const title = await page.title();
const value = await page.evaluate(() => 1 + 1);
const text = (await page.locator("#x").innerText()).trim();
if (title !== "Oracle" || value !== 2 || text !== "ok") {
  throw new Error(JSON.stringify({ title, value, text }));
}
await browser.close();
