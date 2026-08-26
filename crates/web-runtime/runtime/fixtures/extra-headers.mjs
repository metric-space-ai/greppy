import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setExtraHTTPHeaders({ "x-greppy-test": "yes" });
const nav = page.waitForNavigation();
await page.goto(fixtureUrl);
await nav;
await page.mainFrame().goto(fixtureUrl);
const text = (await page.locator("body").innerText()).trim();
if (text !== "HEADER_OK") {
  throw new Error("expected HEADER_OK from extra headers, got " + JSON.stringify(text));
}
const url = await page.waitForURL("http");
if (!String(url).includes("http")) throw new Error("waitForURL " + url);
const request = await page.waitForRequest("http");
if (!String(request.url()).includes("http")) throw new Error("waitForRequest");
if (request.method() !== "GET") throw new Error("method " + request.method());
if (request.resourceType() !== "document") {
  throw new Error("resourceType " + request.resourceType());
}
const headers = request.headers();
const headerBlob = JSON.stringify(headers).toLowerCase();
if (!headerBlob.includes("x-greppy-test")) {
  throw new Error("Request.headers missing extra header: " + JSON.stringify(headers));
}
if (request.headerValue("x-greppy-test") !== "yes") {
  throw new Error("headerValue " + request.headerValue("x-greppy-test"));
}
const headerArray = request.headersArray();
if (!headerArray.some((h) => String(h.name).toLowerCase() === "x-greppy-test" && h.value === "yes")) {
  throw new Error("headersArray " + JSON.stringify(headerArray));
}
if (!request.isNavigationRequest()) throw new Error("isNavigationRequest");
const ctx = await browser.newContext();
if (ctx.browser() !== browser && ctx.browser()._id !== browser._id) {
  throw new Error("context.browser");
}
ctx.setDefaultNavigationTimeout(5_000);
await ctx.route("**/ctx-route", (route) =>
  route.fulfill({
    body: "<!DOCTYPE html><html><body><p id=cr>ctx-ok</p></body></html>",
    contentType: "text/html",
  }),
);
const p2 = await ctx.newPage();
await p2.goto(fixtureUrl + "ctx-route");
if ((await p2.innerText("#cr")).trim() !== "ctx-ok") throw new Error("context.route");
await ctx.close();
let paused = false;
try {
  await page.pause();
  paused = true;
} catch (error) {
  if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
}
if (paused) throw new Error("pause must not succeed as a no-op");
await browser.close();
