import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><input id='q'></body></html>");
await page.locator("#q").click();
await page.keyboard.type("hi");
const value = await page.evaluate(() => document.querySelector("#q").value);
if (value !== "hi") {
  throw new Error("keyboard.type expected hi, got " + JSON.stringify(value));
}
await page.close();
await browser.close();
