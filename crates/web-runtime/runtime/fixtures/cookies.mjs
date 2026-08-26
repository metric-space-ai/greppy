import { chromium } from "playwright";

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
await page.goto(fixtureUrl);
await context.addCookies([{ name: "k", value: "v" }]);
const cookies = await context.cookies();
if (!cookies.some((cookie) => cookie.name === "k" && cookie.value === "v")) {
  throw new Error("expected cookie k=v, got " + JSON.stringify(cookies));
}
await context.clearCookies();
const after = await context.cookies();
if (after.some((cookie) => cookie.name === "k")) {
  throw new Error("clearCookies left k: " + JSON.stringify(after));
}
await context.close();
await browser.close();
