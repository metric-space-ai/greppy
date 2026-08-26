import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(fixtureUrl);
await page.goto(fixtureUrl.replace(/\/?$/, "/two"));
await page.goBack();
const url = await page.url();
if (!url) {
  throw new Error("goBack left an empty URL");
}
await browser.close();
