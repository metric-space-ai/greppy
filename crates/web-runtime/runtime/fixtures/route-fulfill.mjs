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
const recorded = await page.requests();
if (!recorded.length) throw new Error("page.requests empty");
await page.unroute("**/intercepted");
await page.goto(fixtureUrl + "intercepted");
let still = false;
try {
  still = (await page.locator("#x").innerText()).trim() === "intercepted-ok";
} catch (error) {
  still = false;
}
if (still) throw new Error("unroute did not stop fulfill");
if (page.opener() !== null) throw new Error("opener");
await browser.close();
