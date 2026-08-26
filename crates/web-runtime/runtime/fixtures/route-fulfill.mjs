import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.route("**/intercepted", (route) =>
  route.fulfill({
    body: "<!DOCTYPE html><html><body><p id='x'>intercepted-ok</p></body></html>",
    contentType: "text/html",
  }),
);
await page.goto(fixtureUrl + "intercepted");
const text = await page.locator("#x").innerText();
if (text.trim() !== "intercepted-ok") {
  throw new Error("expected intercepted-ok, got " + JSON.stringify(text));
}
await browser.close();
