import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><p>ok</p></body></html>");
if ((await page.locator("p").count()) !== 1) {
  throw new Error("expected one paragraph");
}
await browser.close();
