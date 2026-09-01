import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><input id='q'></body></html>");
if (!(await page.isEditable("#q"))) throw new Error("isEditable");
await page.locator("#q").type("ab");
if ((await page.inputValue("#q")) !== "ab") {
  throw new Error("locator.type expected ab, got " + JSON.stringify(await page.inputValue("#q")));
}
if ((await page.mainFrame().inputValue("#q")) !== "ab") throw new Error("frame inputValue");
if (!(await page.mainFrame().isEditable("#q"))) throw new Error("frame isEditable");
await page.locator("#q").evaluate((el) => {
  el.addEventListener("keydown", () => {
    window.__pressed = 1;
  });
});
await page.locator("#q").press("Enter");
if ((await page.evaluate(() => window.__pressed)) !== 1) {
  throw new Error("locator.press did not dispatch keydown");
}
await browser.close();
