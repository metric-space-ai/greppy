import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  "<!DOCTYPE html><html><body><input type='checkbox' id='c'><select id='s'><option value='a'>A</option><option value='b'>B</option></select></body></html>",
);
await page.locator("#c").check();
await page.locator("#s").selectOption("b");
const state = await page.evaluate(() => ({
  checked: document.querySelector("#c").checked,
  value: document.querySelector("#s").value,
}));
if (!state.checked || state.value !== "b") {
  throw new Error("check/selectOption failed: " + JSON.stringify(state));
}
await page.locator("#c").uncheck();
const unchecked = await page.evaluate(() => document.querySelector("#c").checked);
if (unchecked) {
  throw new Error("uncheck failed");
}
await page.setChecked("#c", true);
if (!(await page.evaluate(() => document.querySelector("#c").checked))) {
  throw new Error("setChecked true failed");
}
await page.setChecked("#c", false);
if (await page.evaluate(() => document.querySelector("#c").checked)) {
  throw new Error("setChecked false failed");
}
await browser.close();
