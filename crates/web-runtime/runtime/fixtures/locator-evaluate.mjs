import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  "<!DOCTYPE html><html><body><input id='q' value='hi'><p id='t'>Hello</p></body></html>",
);
const value = await page.locator("#q").evaluate((el) => el.value);
if (value !== "hi") throw new Error("evaluate value " + JSON.stringify(value));
const tagged = await page.locator("#t").evaluate((el, prefix) => prefix + el.textContent, "x:");
if (tagged !== "x:Hello") throw new Error("evaluate arg " + JSON.stringify(tagged));
const count = await page.locator("body > *").evaluateAll((els) => els.length);
if (count !== 2) throw new Error("evaluateAll " + count);
await browser.close();
