import { chromium } from "playwright";

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
await page.goto(fixtureUrl);
const seenRequest = await page.waitForEvent("request");
if (!seenRequest || typeof seenRequest.url !== "function") {
  throw new Error("waitForEvent request");
}
let httpOnlyFailed = false;
try {
  await context.addCookies([{ name: "h", value: "secret", httpOnly: true }]);
} catch (error) {
  httpOnlyFailed = String(error).includes(
    "unsupported_playwright_operation: BrowserContext.addCookies.httpOnly"
  );
}
if (!httpOnlyFailed) throw new Error("httpOnly cookies must fail closed");
await context.addCookies([{ name: "k", value: "v", path: "/" }]);
const cookies = await context.cookies();
if (!cookies.some((cookie) => cookie.name === "k" && cookie.value === "v")) {
  throw new Error("expected cookie k=v, got " + JSON.stringify(cookies));
}
await page.evaluate(() => localStorage.setItem("greppy", "origin-ok"));
const state = await context.storageState();
if (!state.cookies.some((cookie) => cookie.name === "k" && cookie.value === "v")) {
  throw new Error("storageState missing k=v: " + JSON.stringify(state));
}
if (
  !Array.isArray(state.origins) ||
  !state.origins.some(
    (origin) =>
      origin.localStorage &&
      origin.localStorage.some((item) => item.name === "greppy" && item.value === "origin-ok")
  )
) {
  throw new Error("storageState missing localStorage origin: " + JSON.stringify(state));
}
await context.clearCookies();
const after = await context.cookies();
if (after.some((cookie) => cookie.name === "k")) {
  throw new Error("clearCookies left k: " + JSON.stringify(after));
}
const restoredCtx = await browser.newContext({ storageState: state });
const restoredPage = restoredCtx.pages()[0] || (await restoredCtx.newPage());
await restoredPage.goto(fixtureUrl);
const restoredCookies = await restoredCtx.cookies();
if (!restoredCookies.some((cookie) => cookie.name === "k" && cookie.value === "v")) {
  throw new Error("restored cookies missing k=v: " + JSON.stringify(restoredCookies));
}
const restoredLs = await restoredPage.evaluate(() => localStorage.getItem("greppy"));
if (restoredLs !== "origin-ok") {
  throw new Error("restored localStorage missing greppy: " + JSON.stringify(restoredLs));
}
await restoredCtx.close();
await context.close();
await browser.close();
