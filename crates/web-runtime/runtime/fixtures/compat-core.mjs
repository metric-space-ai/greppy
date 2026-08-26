import { chromium } from "playwright";

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
await page.setContent(
  "<!DOCTYPE html><html><body><h1>Hello</h1><button>Go</button><input id='q'></body></html>",
);
if ((await page.getByText("Hello").count()) < 1) {
  throw new Error("getByText did not see Hello");
}
if (!(await page.getByText("Hello").isVisible())) {
  throw new Error("Hello should be visible");
}
await page.getByText("Go").hover();
await page.getByText("Go").waitFor();
await page.locator("input").first().fill("ok");
await page.evaluate(() => document.title);
await page.waitForTimeout(10);
await page.waitForLoadState();
await browser.close();
