import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><p id='x'>ready</p></body></html>");
await page.waitForSelector("#x");
const text = (await page.locator("#x").innerText()).trim();
if (text !== "ready") {
  throw new Error("waitForSelector expected ready, got " + JSON.stringify(text));
}
await browser.close();
