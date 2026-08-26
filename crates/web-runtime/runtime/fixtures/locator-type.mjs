import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><input id='q'></body></html>");
if (!(await page.isEditable("#q"))) throw new Error("isEditable");
await page.locator("#q").type("ab");
if ((await page.inputValue("#q")) !== "ab") {
  throw new Error("locator.type expected ab, got " + JSON.stringify(await page.inputValue("#q")));
}
await browser.close();
