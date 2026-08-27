import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><button id='b'>Go</button></body></html>");
await page.locator("#b").hover({ timeout: 5_000 });
await page.hover("#b", { timeout: 5_000 });
await page.locator("#b").click({ timeout: 5_000 });
await browser.close();
