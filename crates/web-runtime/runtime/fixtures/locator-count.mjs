import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><li>a</li><li>b</li></body></html>");
const count = await page.locator("li").count();
if (count !== 2) {
  throw new Error("expected 2 list items, got " + count);
}
if (!(await page.locator("li").nth(0).isVisible())) {
  throw new Error("first list item should be visible");
}
await browser.close();
