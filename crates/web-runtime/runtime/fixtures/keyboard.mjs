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
await page.keyboard.insertText("!");
const inserted = await page.evaluate(() => document.querySelector("#q").value);
if (inserted !== "hi!") {
  throw new Error("insertText expected hi!, got " + JSON.stringify(inserted));
}
await page.evaluate(() => {
  window.__keys = [];
  document.getElementById("q").addEventListener("keydown", (event) => {
    window.__keys.push(event.key);
  });
});
await page.keyboard.press("Enter");
const keys = await page.evaluate(() => window.__keys);
if (!Array.isArray(keys) || !keys.includes("Enter")) {
  throw new Error("keyboard.press Enter, got " + JSON.stringify(keys));
}
await page.close();
await browser.close();